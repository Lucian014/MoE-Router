"""
memory/indexer.py -- index one stored turn into chunks (docs/rag-planv2.md 2, steps 9-10).

    turn (prompt + answer) -> header -> chunks -> embeddings + BM25 terms -> `chunks` rows

The embeddings are computed outside the database transaction; the header and the chunks are then
written in ONE transaction (W2), so a turn is either fully indexed or not at all. Idempotent: a
turn already indexed with the current chunker and embedding model is skipped, so a redelivered
message changes nothing; a turn indexed with older rules is replaced (reindexing).
"""
from dataclasses import dataclass

from database import ChunkRepository, TurnRepository, get_session
from shared.memory.analyzer import analyze
from shared.memory.chunking import CHUNKER_VERSION, chunk_turn, embedding_input, terms_input
from shared.memory.embedding import EmbeddingClient
from shared.memory.header import extract_question


@dataclass(frozen=True)
class IndexResult:
    turn_id: int
    status: str          # indexed | reindexed | skipped (up to date) | missing (no such turn)
    chunks: int = 0


async def index_turn(turn_id: int, embedder: EmbeddingClient, force: bool = False) -> IndexResult:
    async with get_session() as session:
        turn = await TurnRepository(session).get(turn_id)
        existing = await ChunkRepository(session).list_for_turn(turn_id) if turn else []
    if turn is None:
        return IndexResult(turn_id, "missing")
    up_to_date = existing and all(
        c.chunker_version == CHUNKER_VERSION and c.embedding_model == embedder.model for c in existing)
    if up_to_date and not force:
        return IndexResult(turn_id, "skipped", len(existing))

    header, _ = extract_question(turn.user_content, turn.assistant_content)
    drafts = chunk_turn(turn.user_content, turn.assistant_content)
    vectors = await embedder.embed([embedding_input(header, d.content) for d in drafts])
    rows = [
        dict(turn_id=turn.id, session_id=turn.session_id, turn=turn.turn, ord=d.ord, source=d.source,
             kind=d.kind, lang=d.lang, content=d.content, content_hash=d.content_hash, tokens=d.tokens,
             terms=analyze(terms_input(header, d.content)), embedding=vector,
             embedding_model=embedder.model, chunker_version=CHUNKER_VERSION)
        for d, vector in zip(drafts, vectors)
    ]

    async with get_session() as session:            # W2: header + chunks, all or nothing
        chunks = ChunkRepository(session)
        if existing:
            await chunks.delete_for_turn(turn.id)
        await TurnRepository(session).set_header(turn.id, header)
        await chunks.insert_many(rows)
    return IndexResult(turn_id, "reindexed" if existing else "indexed", len(rows))
