"""
memory/indexer.py -- index one stored turn into chunks (docs/rag-planv2.md 2, steps 9-10).

    turn (prompt + answer) -> header -> chunks -> embeddings + BM25 terms -> `chunks` rows

The embeddings are computed outside the database transaction; the header and the chunks are then
written in ONE transaction (W2), so a turn is either fully indexed or not at all. Idempotent: a
turn already indexed with the current rules (chunking.index_version) and embedding model is
skipped, so a redelivered message changes nothing; a turn indexed with older rules is replaced
(reindexing) -- or, when only the BM25 analyzer changed, keeps its chunks and vectors and gets
its terms recomputed (no embedding call).
"""
from dataclasses import dataclass

from database import ChunkRepository, TurnRepository, get_session
from shared.memory.analyzer import analyze
from shared.memory.chunking import chunk_turn, embedding_input, index_version, same_chunks, terms_input
from shared.memory.embedding import EmbeddingClient
from shared.memory.header import extract_question


@dataclass(frozen=True)
class IndexResult:
    turn_id: int
    status: str          # indexed | reindexed | reanalyzed (terms only) | skipped (up to date) | missing
    chunks: int = 0


async def index_turn(turn_id: int, embedder: EmbeddingClient, force: bool = False) -> IndexResult:
    async with get_session() as session:
        turn = await TurnRepository(session).get(turn_id)
        existing = await ChunkRepository(session).list_for_turn(turn_id) if turn else []
    if turn is None:
        return IndexResult(turn_id, "missing")
    version = index_version()
    same_model = existing and all(c.embedding_model == embedder.model for c in existing)
    if not force and same_model and all(c.chunker_version == version for c in existing):
        return IndexResult(turn_id, "skipped", len(existing))

    header, from_answer = extract_question(turn.user_content, turn.assistant_content)
    if not force and same_model and all(same_chunks(c.chunker_version) for c in existing):
        async with get_session() as session:
            chunks = ChunkRepository(session)
            for c in existing:
                await chunks.set_terms(c.id, analyze(terms_input(header, c.content)), version)
        return IndexResult(turn_id, "reanalyzed", len(existing))

    drafts = chunk_turn(turn.user_content, turn.assistant_content)
    vectors = await embedder.embed([embedding_input(header, d.content) for d in drafts])
    rows = [
        dict(turn_id=turn.id, session_id=turn.session_id, turn=turn.turn, ord=d.ord, source=d.source,
             kind=d.kind, lang=d.lang, content=d.content, content_hash=d.content_hash, tokens=d.tokens,
             terms=analyze(terms_input(header, d.content)), embedding=vector,
             embedding_model=embedder.model, chunker_version=version)
        for d, vector in zip(drafts, vectors)
    ]

    async with get_session() as session:
        chunks = ChunkRepository(session)
        if existing:
            await chunks.delete_for_turn(turn.id)
        await TurnRepository(session).set_header(turn.id, header, from_answer)
        await chunks.insert_many(rows)
    return IndexResult(turn_id, "reindexed" if existing else "indexed", len(rows))
