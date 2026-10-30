# Plan RAG pentru contextul comun al sesiunilor

Adaptarea *Contextual Retrieval* (Anthropic, 2024) la MoE-Router: memorie de conversație comună tuturor modelelor, Ollama pe CPU (vLLM pe GPU mai târziu), Postgres + pgvector, indexare asincronă. Mai multe versiuni de RAG, comparate pe același set de date, pentru evaluarea din licență.

---

## 1. Decizii de design

- **Unitatea de memorie e tura** (prompt + răspuns). „Documentul” din articol = tura; chunk-urile = bucățile răspunsului și ale conținutului lipit în prompt.
- **Chunking-ul depinde de structura textului, nu de categorie.** Un singur chunker, cu câte o strategie per tip de segment (proză, cod, log, formule, liste, tabele). Categoria se folosește doar la raportarea rezultatelor.
- **Starea stă doar în Postgres.** Toate serviciile rămân stateless și se scalează prin replici.
- **Un scriitor per tabelă.** Worker-ul scrie `turns` și `sessions` sincron, înainte de răspuns. Serviciul `memory` scrie `chunks`, asincron. Routerul doar citește.
- **Clasificatorul vede doar prompt-ul curent.** Dacă rezultatul e ambiguu, sesiunea păstrează domeniul și complexitatea turei anterioare.
- **Contextul se construiește după alegerea modelului**, în bugetul lui de tokeni. Materialul (ture recente, candidați RAG) se adună înainte, în paralel cu clasificarea.
- **Numărul de ture recente rezultă din buget, nu e fix.** Se citesc ultimele 20 de ture, iar câte intră întregi depinde de modelul ales. Un model mic primește 2–3 ture și fragmente, unul mare poate primi toate cele 20. RAG-ul acoperă ce nu încape.
- **Routerul nu ține nimic în memorie între request-uri.** Toate turele stau permanent în `turns`, iar la fiecare request se citesc din Postgres doar cele necesare.
- **Indexezi o dată, evaluezi toate versiunile.** Toate reprezentările unui chunk stau pe același rând, iar versiunea se alege prin configurare.
- **Prompt caching doar la contextualizare.** La generare nu se folosește: fragmentele, fereastra glisantă, schimbarea modelului și replicile strică prefixul comun.
- **Fără rezumat al sesiunii și fără Redis.** Pe CPU, rezumatul concurează cu userii. Citirea sesiunii e un query de câteva ms. Ambele se reevaluează după măsurători sau după trecerea pe GPU.

---

## 2. Fluxul

**Sincron** (cât userul așteaptă):

1. Gateway: `POST /generate {prompt, session_id?}`. Generează `session_id` dacă lipsește și îl întoarce în răspuns.
2. Router, citirea R1: `sessions` (starea de rutare) + ultimele `RAG_RECENT_TURNS_MAX` (20) ture, întregi. Tura curentă = ultima + 1. Un `session_id` necunoscut înseamnă sesiune goală.
3. Router, în paralel (`asyncio.gather`):
   - **rutare de sesiune** (4.1) → domeniu și complexitate efective
   - **retrieval**, dacă sesiunea are cel puțin 2 ture: query-urile (4.4, 4.5), apoi R2 pe chunk-urile **tuturor turelor, mai puțin ultima**, apoi dense + BM25 → RRF → deduplicare → `RAG_POOL` candidați (4.6)
4. Router: `select_model` → `context_window`, `max_output_tokens` → turele recente care încap → filtrarea candidaților → R3 (conținutul fragmentelor alese) → asamblare (4.7).
5. Router → worker: `{messages, session_id, turn, routing, retrieved_chunk_ids}`.
6. Worker: `engine.chat(messages)` → răspuns.
7. Worker, scrierea W1, într-o tranzacție: `INSERT sessions ON CONFLICT DO NOTHING`, `INSERT turns`, `UPDATE sessions` (starea de rutare efectivă).
8. Worker: răspunde gateway-ului, apoi publică `{turn_id}` pe `memory.ingest`.

**Asincron** (serviciul `memory`):

9. Citește tura, extrage header-ul, face chunking, embedding batch (bge-m3) și termeni BM25.
10. Scrierea W2: `INSERT chunks ON CONFLICT DO NOTHING`, apoi ack. Tura devine regăsibilă prin RAG.

O tură care încape în bugetul modelului ales intră integral în prompt. Altfel intră doar prin chunk-urile ei relevante. Aceeași tură poate fi întreagă pe un model mare și doar fragmente pe unul mic.

**Accesări ale DB per request:** R1 (sesiunea + ultimele 20 de ture), R2 (distanța și termenii chunk-urilor eligibile, fără vectori și fără conținut), R3 (conținutul fragmentelor alese), W1 (tura), plus W2 asincron.

**Robustețea `memory`:** ack după commit, idempotență prin `UNIQUE (session_id, turn, ord)`, DLQ pentru mesajele care eșuează repetat. Dacă publicarea din pasul 8 se pierde, tura există oricum în `turns`, iar un job de reindexare tratează turele fără chunk-uri.

---

## 3. Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;   -- imaginea pgvector/pgvector:pg16

CREATE TABLE sessions (
    id                uuid PRIMARY KEY,
    user_id           text,
    domain            text,          -- starea de rutare efectivă
    domain_confidence real,
    complexity_score  real,
    created_at        timestamptz NOT NULL DEFAULT now(),
    last_activity_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE turns (
    id                   bigserial PRIMARY KEY,
    session_id           uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn                 int  NOT NULL,
    user_content         text NOT NULL,
    assistant_content    text NOT NULL,
    header               text,          -- completat de memory
    model_id             text,
    predicted_domain     text,  predicted_complexity real,
    effective_domain     text,  effective_complexity real,
    inherited            boolean NOT NULL DEFAULT false,
    retrieved_chunk_ids  bigint[],      -- log de diagnostic
    created_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn)
);

CREATE TABLE chunks (
    id               bigserial PRIMARY KEY,
    turn_id          bigint NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    session_id       uuid   NOT NULL,   -- copiat, ca filtrul să nu ceară JOIN
    turn             int    NOT NULL,
    ord              int    NOT NULL,
    source           text   NOT NULL CHECK (source IN ('prompt', 'response')),
    kind             text   NOT NULL CHECK (kind IN ('prose', 'code', 'log', 'math', 'list', 'table')),
    lang             text,                  -- limbajul, pentru kind = 'code'
    content          text   NOT NULL,
    content_hash     text   NOT NULL,       -- sha256 pe conținutul normalizat (4.6)
    tokens           int    NOT NULL,
    terms            text[] NOT NULL,       -- V1/V2: BM25 pe header + content
    embedding        vector(1024) NOT NULL, -- V1/V2: "Întrebare: {header} | {content}"
    context          text,                  -- V3+: NULL până la contextualizare
    terms_ctx        text[],
    embedding_ctx    vector(1024),          -- V3+: "{context}\n{content}"
    embedding_model  text NOT NULL,
    context_model    text,
    chunker_version  text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, turn, ord)
);
CREATE INDEX chunks_session_turn ON chunks (session_id, turn);

ALTER TABLE models ADD COLUMN context_window    int NOT NULL DEFAULT 4096;
ALTER TABLE models ADD COLUMN max_output_tokens int NOT NULL DEFAULT 512;
```

Fără HNSW: căutarea e mereu pe o singură sesiune, câteva sute de rânduri, deci comparația exactă e rapidă și deterministă.

`context_window` = valoarea **efectivă** de rulare (`num_ctx` la Ollama, `--max-model-len` la vLLM), nu maximul modelului. Include intrarea și ieșirea. Se trimite la fiecare request.

**Configurare** (`RetrievalSettings` în `src/shared/config.py`):

```
RAG_DENSE / RAG_BM25 / RAG_CONTEXTUAL / RAG_RERANK = true|false
RAG_CANDIDATES=50             # per metodă, înainte de fuziune sau reranking
RAG_POOL=30                   # candidați păstrați după fuziune, înainte de filtrarea după model
RAG_TOP_K=10                  # fragmente maxime după filtrare
RAG_RECENT_TURNS_MAX=20       # câte ture citește R1; plafonul turelor întregi
RAG_RECENT_MAX_SHARE=0.6      # cât din buget pot ocupa turele recente; restul e pentru fragmente
RAG_LAST_TURN_MAX_SHARE=0.5   # cât poate ocupa ultima tură înainte de trunchiere
RAG_MAX_CONTEXT_TOKENS=0      # experiment: plafon de context indiferent de model (0 = dezactivat)
RAG_MAX_CHUNKS_PER_TURN=0     # 0 = fără limită
RAG_DEDUP=off|hash|near       # deduplicarea candidaților (4.6)
RAG_DEDUP_JACCARD=0.8         # pragul pentru aproape-duplicate
RAG_DENSE_QUERY_MAX_TOKENS=512  # query-ul dense pe prompt-uri fără proză (4.4)
```

---

## 4. Componente

### 4.1 Rutarea de sesiune

1. **Prompt scurt sau de continuare** (≤ 3 cuvinte sau o listă de expresii: „continuă”, „ok”, „și în …”) și sesiunea are stare → moștenește domeniul și complexitatea, fără clasificator.
2. Altfel clasifică **doar prompt-ul curent**.
3. **Domeniu ambiguu** (probabilitate maximă < 0.5 sau diferența dintre primele două < 0.15) → moștenește **ambele** valori.
4. Prima tură a sesiunii → rezultatul clasificatorului, fără moștenire.

Pragurile stau în `ClassifierSettings`. Valorile prezise, cele efective și `inherited` se salvează pe tură.

Motivul: clasificatoarele au fost antrenate pe prompt-uri individuale, iar pe „continua” clasificatorul de domeniu a dat `summarization` cu 0.91. Regula 1 acoperă cazurile în care clasificatorul e greșit cu încredere mare, regula 3 pe cele în care distribuția e plată.

### 4.2 Header-ul turei (`extract_question`)

Text lipit în fața chunk-ului înainte de embedding și BM25. Un singur vector per chunk.

1. Prompt-ul se separă în proză și conținut lipit (cod, log-uri), cu segmentarea din 4.3.
2. Proza are ≤ 150 de tokeni → proza întreagă. Așa, majoritatea prompt-urilor trec neschimbate.
3. Proza are > 150 de tokeni → întâi propozițiile cu `?`, apoi cele de la final spre început, până la ~150 de tokeni, păstrate în ordinea originală.
4. Fără proză → prima propoziție din răspuns.

Header-ul se pune pe toate chunk-urile turei. Proza scurtă a userului (≤ 150 de tokeni) nu produce chunk-uri proprii, pentru că e deja header-ul. Proza lungă produce și chunk-uri (4.3). Dacă top-k e dominat de o singură tură, se activează `RAG_MAX_CHUNKS_PER_TURN`.

Header-ul **nu** intră în `chunks.content`. Se salvează o dată, în `turns.header`, și se folosește:

- la indexare, în textul din care se calculează `embedding` și `terms`
- la asamblare, ca **titlu al grupului de fragmente** din tura respectivă (4.7), ca modelul să știe la ce cerere răspundeau fragmentele. Dacă header-ul vine din răspuns (pasul 4), titlul e marcat `Subiect:` în loc de `Întrebare:`, ca modelul să nu-l atribuie userului.

### 4.3 Chunking

**Principiul: tăierea depinde de structura textului, nu de categoria conversației.** Un singur chunker, cu câte o strategie pentru fiecare **tip de segment**. Categoria nu se folosește la chunking, pentru că:

- e o predicție pe prompt, poate fi greșită sau moștenită, iar chunking-ul se face pe răspuns
- turele amestecă tipuri de conținut (o tură de `math` cu cod Python, una de `coding` cu proză explicativă)
- proza se taie la fel în toate categoriile, deci regulile per categorie s-ar duplica
- un singur `chunker_version` descrie complet tăierea, deci diferențele dintre categorii la evaluare vin din RAG, nu din chunker

Categoria se folosește doar la evaluare, pentru rezultatele per categorie (`turns.effective_domain`).

**Ce se face chunk-uri:**

| Partea turei | Rezultat |
|---|---|
| proza userului ≤ 150 de tokeni | doar header (4.2) |
| proza userului > 150 de tokeni (un articol de rezumat, un text de tradus sau de reformulat) | header (selecția) **și** chunk-uri, `source = 'prompt'` |
| cod, log, formule, tabele lipite de user | chunk-uri, `source = 'prompt'` |
| răspunsul asistentului | chunk-uri, `source = 'response'` |

Fără regula pentru proza lungă, la `summarization`, `translation` și `text-rewrite` textul original n-ar fi indexat, iar o întrebare ulterioară despre el n-ar avea ce găsi.

**Pasul 1: segmentarea** (`segment(text) -> list[Segment]`, cu `kind`, `text` și `lang` pentru cod):

| Tip | Cum se recunoaște |
|---|---|
| `code` | fence-uri ` ``` ` și `~~~`, cu o mașină de stări: închiderea cere același tip și cel puțin aceeași lungime, un fence neînchis (răspuns tăiat la `max_tokens`) e cod până la final. Fără fence: scor per linie (indentare, simboluri, cuvinte-cheie), grupuri de ≥ 3 linii |
| `log` | tipare de log și traceback (`Traceback`, `File "...", line`, `...Error:`, timestamp-uri), grupuri de ≥ 3 linii |
| `math` | blocuri `$$ ... $$` și `\[ ... \]` |
| `table` | tabele markdown (linia de separare `\|---\|`) |
| `list` | elemente consecutive `-`, `*`, `1.` |
| `prose` | restul, cu titlurile markdown păstrate ca limite de secțiune |

Detectarea codului și a log-urilor fără fence se aplică pe prompt **și**, ca rezervă, pe proza din răspuns, pentru modelele mici care uită uneori fence-urile. System prompt-ul cere oricum explicit codul în blocuri ` ``` ` (4.7). Un bloc de cod nerecunoscut ajunge în `prose`, tăiat pe paragrafe. Căutarea funcționează la fel, iar pierderea e doar tăierea precisă cu `ast`.

**Pasul 2: tăierea**, doar pentru segmentele de peste `MAX_TOKENS=400`. Cele mai mici rămân întregi. Fiecare tip are un `Splitter` (`split(segment, max_tokens) -> list[str]`):

| Tip | Splitter |
|---|---|
| `prose` | titluri, apoi paragrafe, apoi propoziții |
| `code` | Python: `ast` pe nodurile de nivel superior, decoratorii cu funcția, clasele mari pe metode cu semnătura clasei în față. La `SyntaxError` și pentru alte limbaje: linii goale, apoi împachetare, apoi tăiere brută |
| `log` | pe linii, păstrând finalul împreună (excepția și ultimele cadre) |
| `list` | între elemente, niciodată în interiorul unui element |
| `math`, `table` | atomic: nu se taie, chiar dacă depășesc ușor limita |

Un tip nou (de ex. diagrame Mermaid) înseamnă un splitter nou, fără modificări în rest.

**Pasul 3: normalizarea.** Bucățile sub `MIN_TOKENS=50` se lipesc de următoarea (ultima de precedenta). O bucată lipită primește tipul ne-proză cel mai mare din ea. Efectul dorit: „Iată funcția:” ajunge împreună cu codul pe care îl introduce. `MAX_TOKENS` e o țintă, nu o limită absolută.
- **`count_tokens`**: un singur tokenizer de referință (familia Qwen, prin `tokenizers`), folosit la chunking, `tokens` și buget.
- **`chunker_version`**: se schimbă la orice modificare a constantelor. Invariant testat: concatenarea chunk-urilor reproduce textul original.

### 4.4 Embedding-uri

bge-m3 (1024 dimensiuni, multilingv), prin Ollama `/api/embed`, cu un apel batch per tură.

**Query-ul dense** (`build_dense_query`):

1. Prompt-ul are proză → aceleași reguli ca header-ul (4.2, pașii 1–3). Un traceback lipit lângă o întrebare nu diluează vectorul.
2. Prompt-ul n-are proză (doar o eroare, un log sau cod) → tot prompt-ul dacă are ≤ `RAG_DENSE_QUERY_MAX_TOKENS` (512), altfel **ultimii** 512 tokeni. La un traceback, excepția și cadrul în care a apărut sunt la final, iar începutul e partea cea mai puțin utilă.

Trunchierea se face în cod, cu `count_tokens`, înainte de apel. Ollama trunchiază automat textele prea lungi, dar de la final, adică exact excepția. Comportamentul parametrului `truncate` se verifică pe versiunea folosită.

Risc pe CPU: bge-m3 (~1.2 GB) stă pe același Ollama ca modelele de generare și e apelat la fiecare request. Se setează `keep_alive` și `OLLAMA_MAX_LOADED_MODELS` astfel încât să nu scoată modele de generare din memorie, iar latența se măsoară. Alternativa e o instanță Ollama separată.

### 4.5 BM25

În Python, per sesiune, pe `terms` precalculate. IDF-ul se calculează pe chunk-urile eligibile. Query-ul BM25 = **tot prompt-ul**, inclusiv codul sau traceback-ul lipit.

Analizorul (comun pentru indexare și căutare): litere mici, diacritice normalizate, `snowballstemmer` pentru română, identificatorii păstrați și despărțiți (`create_access_token` → `create_access_token`, `create`, `access`, `token`). Comenzile LaTeX (`\frac`, `\sum`, acolade) se ignoră, iar variabilele și cuvintele din formule rămân.

`ts_rank` din Postgres nu e BM25 și nu se folosește.

### 4.6 Fuziune și reranking

R2 e un singur query pe chunk-urile sesiunii, fără ultima tură: întoarce `id, turn, ord, tokens, terms, content_hash` și distanța `embedding <=> :q`, fără vectori și fără conținut. Clasamentul dense și BM25 se fac în Python pe același rezultat.

RRF: `scor = Σ 1 / (60 + rang)` peste top `RAG_CANDIDATES` din fiecare metodă, apoi deduplicarea id-urilor (același chunk din ambele liste), apoi deduplicarea de conținut, apoi top `RAG_POOL`. Lista e mai lungă decât `RAG_TOP_K` pentru că o parte din candidați se elimină după alegerea modelului (4.7).

**Deduplicarea de conținut.** Contextul asamblat nu se salvează niciodată: pe tură se salvează doar prompt-ul original și răspunsul. Duplicatele apar din conținutul turelor. Modelul rescrie codul întreg cu o modificare, sau userul relipește același fișier ori traceback. Turele sunt legitime și se păstrează, iar problema se tratează la căutare, unde versiunea veche nu trebuie să ajungă la model în locul celei noi.

- `hash`: dintre candidații cu același `content_hash`, rămâne cel din tura cea mai nouă. Normalizarea dinaintea hash-ului: spațiile de la final de linie eliminate, terminatorii de linie unificați, liniile goale consecutive comprimate.
- `near`: `hash`, plus eliminarea unui candidat a cărui similaritate Jaccard pe `terms` față de un candidat **mai nou**, deja păstrat, depășește `RAG_DEDUP_JACCARD`. Termenii sunt deja în memorie pentru BM25, iar vectorii nu ies din DB. E o variantă simplă de MMR (*Maximal Marginal Relevance*).
- La scor RRF egal câștigă tura mai nouă.

Reranking (doar V5): un cross-encoder multilingv (familia `bge-reranker`), pe 50 → top-k. Adaugă latență la fiecare request, iar costul se măsoară explicit.

### 4.7 Asamblarea în buget

```
disponibil = context_window − max_output_tokens − system − prompt curent − 10%
disponibil = min(disponibil, RAG_MAX_CONTEXT_TOKENS)      # doar dacă experimentul e activ
```

`build_messages` e o funcție pură, cu pașii:

1. **Ultima tură**, garantată. Dacă depășește `RAG_LAST_TURN_MAX_SHARE × disponibil`, răspunsul se trunchiază cu `[…]`.
2. **Turele recente**, din cele citite de R1, de la cea mai nouă spre cea mai veche, doar întregi, cât timp totalul rămâne sub `RAG_RECENT_MAX_SHARE × disponibil`. Cea mai veche tură inclusă e **granița**.
3. **Filtrarea candidaților:** se elimină chunk-urile din turele incluse întregi și cele aproape identice (Jaccard pe `terms`) cu chunk-uri din turele incluse, de ex. o versiune veche a unei funcții rescrise recent. Apoi `RAG_MAX_CHUNKS_PER_TURN` și top `RAG_TOP_K`.
4. **R3:** conținutul pentru candidații rămași.
5. **Fragmentele**, în ordinea RRF, în bugetul rămas (`disponibil` − turele recente). Costul unei ture regăsite = header-ul ei (o dată per tură) + chunk-urile alese. Un chunk care nu încape se sare.
6. **Spațiul rămas** (puțini candidați, sesiune scurtă): se adaugă ture mai vechi întregi, din cele citite de R1, cât încap. Fragmentele din turele adăugate se scot, iar spațiul eliberat se refolosește.

Consecințe:

- **Model mare, sesiune ≤ 20 de ture care încape:** toate turele întregi, fără fragmente.
- **Model mare, sesiune lungă:** ultimele 20 de ture întregi, plus fragmente din turele mai vechi.
- **Model mic:** câteva ture întregi, plus fragmente din tot restul sesiunii.

Dacă prompt-ul curent singur depășește bugetul: model cu context mai mare, altfel eroare 413.

Ordinea mesajelor, cronologică:

```
system      instrucțiuni: „răspunde în limba utilizatorului”,
            „pune orice cod în blocuri ``` cu numele limbajului”
system      fragmente regăsite, grupate pe tură:
              [tura N] Întrebare: {header}      (sau Subiect: {header}, 4.2)
                {chunk}  [...]  {chunk}         ([...] între bucăți neconsecutive)
user/asst   turele recente, în ordine cronologică
user        prompt-ul curent
```

Pe tură se salvează numărul de ture întregi, id-urile fragmentelor și top-k-ul efectiv. Pe modelele mici, top-k-ul efectiv e mai mic decât `RAG_TOP_K`.

### 4.8 Contextualizarea (V3–V5)

Prompt-ul Anthropic adaptat: `<document>` (tura întreagă) **primul**, `<chunk>` la final, și instrucțiunea „1–2 propoziții, în limba conversației, doar contextul”. Model local mic (`qwen2.5:3b` / `gemma3:4b`).

Pe CPU rulează **doar offline, în harness**. În cluster ar concura cu userii, deci V3–V5 nu se integrează pe infrastructura actuală.

**Prompt caching:** prefixul (documentul) e identic pentru toate chunk-urile turei, pe același model, în apeluri consecutive. Experimentul compară trei variante: document primul cu cache, document primul fără cache (prefix unic per apel), chunk primul. Se măsoară `prompt_eval_count`, `prompt_eval_duration` și timpul per tură.

---

## 5. Versiuni

| | Dense | BM25 | Contextual | Rerank |
|---|---|---|---|---|
| V0: doar ture recente | | | | |
| V1: RAG simplu | ✓ | | | |
| V2: hibrid | ✓ | ✓ | | |
| V3: contextual dense | ✓ | | ✓ | |
| V4: contextual hibrid | ✓ | ✓ | ✓ | |
| V5: complet (opțional) | ✓ | ✓ | ✓ | ✓ |

Versiunile sunt **preseturi cu nume** peste `RetrievalSettings`, definite într-un singur loc (`src/shared/memory/presets.py`). Același cod rulează toate versiunile, doar cu altă configurare.

Axe transversale: top-k ∈ {5, 10, 20}, prompt caching on/off (V3–V5, cost de indexare).

Ablații: fără header (V1a) vs header extras vs trunchiere naivă; query dense pe prompt întreg vs întrebarea extrasă; pe prompt-urile fără proză, `RAG_DENSE_QUERY_MAX_TOKENS` 256 / 512 / 1024 vs fără dense (doar BM25); `RAG_MAX_CHUNKS_PER_TURN` 0 vs 2; `RAG_RECENT_MAX_SHARE` 0.4 / 0.6 / 0.8; număr de ture recente fix (4) vs rezultat din buget; `RAG_DEDUP` off / hash / near; header ca titlu de grup on/off; moștenirea în rutare on/off.

Experiment pe modelul mare: `RAG_MAX_CONTEXT_TOKENS` ∈ {4000, 8000, 16000, dezactivat}. Compară calitatea, latența până la primul token și tokenii procesați. Ipoteza: un context limitat + RAG dă răspunsuri comparabile cu tot contextul, cu cost mult mai mic.

Ipoteza de verificat: pe CPU, V2 are cel mai bun raport calitate/cost, iar contextualizarea aduce puțin peste header.

---

## 6. Evaluare

**Set de date:** 20–40 de sesiuni a câte 10–20 de ture, generate cu un model mare și verificate manual, din **toate categoriile** catalogului, nu doar coding. Sesiunile de `summarization`, `translation` și `text-rewrite` conțin texte lungi lipite de user, iar cele de `math` conțin formule LaTeX. 5–10 întrebări de test per sesiune, despre ture **vechi**: din afara ultimelor 20 sau care nu încap întregi pe modelul mic. Câteva sesiuni trebuie să aibă peste 20 de ture, ca RAG-ul să fie testat și pe modelul mare. Tipuri: semantice, cu termeni exacți, ambigue, cu cod sau log lipit înainte, doar o eroare sau un traceback (fără text, cu gold pe tura care conținea codul cu problema), RO și EN. Turele de bază includ cod lipit înaintea întrebării și prompt-uri doar cu log. Unele sesiuni conțin cod rescris de mai multe ori, cu întrebări despre versiunea curentă: gold e citatul din **ultima** versiune, iar un hit pe o versiune veche se numără separat, ca „versiune depășită în top-k”.

**Etichete gold:** tura + un **citat** din text, nu id-uri de chunk. Harness-ul mapează citatul pe chunk-uri, deci etichetele rămân valide când se schimbă `chunker_version`. Hit = cel puțin un chunk gold în top-k.

**Metrici:**
- căutare: Recall@k, MRR, ture distincte în top-k, versiuni depășite în top-k
- **prezența în context**, per model: informația gold a ajuns la model, fie pentru că tura gold a intrat întreagă, fie pentru că un chunk gold a intrat ca fragment. Recall@k măsoară căutarea, independent de model; prezența în context măsoară căutarea + bugetul
- cap-coadă: LLM-judecător 1–5 (verificat manual pe un eșantion), tokeni de context, ture întregi vs fragmente, top-k efectiv
- formatare: procentul de răspunsuri cu fence-uri corecte, per model
- **toate metricile de căutare și cap-coadă se raportează și per categorie** (`effective_domain`) și per tip de chunk găsit (`kind`), ca să se vadă unde RAG-ul funcționează slab
- latență pe etape: R1, embedding, dense, BM25, fuziune, reranking, generare
- indexare: timp per tură, apeluri LLM per tură
- rutare: acuratețe pe conversații de mai multe ture, cu și fără moștenire

**Harness offline** (principalul mod de evaluare): apelează direct funcțiile din `src/shared/memory/`, fără RabbitMQ sau cluster. Indexează o dată toate reprezentările și rulează preset × top-k × ablație × model (mic / mare, prin `context_window`). Rezultatele includ lungimea sesiunii și tipul întrebării.

**Cap-coadă în cluster**, pentru 2–3 preseturi finale: `/generate` acceptă un câmp `variant` (numele presetului), luat în considerare doar dacă `ALLOW_VARIANT_OVERRIDE=true`, setat în mediul de test. Varianta se alege **per sesiune**: harness-ul trimite toate turele unei conversații cu aceeași variantă. Varianta se salvează pe tură, în `routing`. Routere separate, fiecare cu coada lui, sunt necesare doar pentru diferențe care nu se pot schimba per request (alt checkpoint de clasificator). Două routere diferite nu se pun niciodată pe aceeași coadă.

---

## 7. Plan de implementare

Fiecare pas se termină cu teste și un criteriu clar de „gata”. Pașii 1–4 dau context comun funcțional în cluster (V0), înainte de orice RAG.

### Pasul 1: Migrarea și catalogul

- **Postgres:** imaginea devine `pgvector/pgvector:pg16` în `charts/moe-router/values.yaml` (`postgres.image`). Verifici că datele existente rămân după upgrade (aceeași versiune majoră).
- **Migrare Alembic** în `src/database/migrations/versions/`: extensia `vector`, `sessions`, `turns`, `chunks`, cele două coloane noi pe `models`. Fără autogenerate pentru `vector(1024)`: tipul se scrie manual cu `pgvector.sqlalchemy.Vector`.
- **Modele ORM** în `src/database/models/`: `session/`, `turn/`, `chunk/`, câte un `schema.py` și un `repository.py`, după modelul `model_registry/`.
- **Catalog:** `context_window` și `max_output_tokens` în `src/catalog/models.yaml`, citite de `scripts/seed_models.py`. Valoarea = `num_ctx` pe care o setezi efectiv.
- **Gata când:** `dbctl.sh upgrade head`, `downgrade -1` și din nou `upgrade head` trec, iar `skaffold run` aplică migrarea prin Job-ul de setup.

### Pasul 2: `chat()` cu mesaje în motoare

- `src/engine/base.py`: `generate(model, prompt)` devine `chat(model, messages, max_tokens, num_ctx) -> ChatResult`, cu `text`, `prompt_tokens`, `completion_tokens`.
- `ollama_engine.py`: `/api/chat` cu `options.num_ctx` și `options.num_predict`. `prompt_tokens` vine din `prompt_eval_count`.
- `vllm_engine.py`: `/v1/chat/completions`. `num_ctx` se ignoră, pentru că e fixat de `--max-model-len`.
- `src/worker/worker.py`: consumă `messages` și actualizează `_CONSUMED_FIELDS`. Ca tranziție, acceptă temporar și `prompt` simplu.
- **Gata când:** testele unitare cu `httpx` mock-uit trec pe ambele motoare, iar un request prin cluster răspunde în limba prompt-ului, datorită system prompt-ului.

### Pasul 3: Sesiuni și scrierea sincronă a turei

- **Gateway** (`src/gateway/main.py`): `/generate` primește `session_id` opțional, generează un uuid dacă lipsește și îl întoarce. Gateway-ul rămâne fără DB.
- **Worker:** după `chat()`, face W1 într-o tranzacție (`sessions` upsert, `turns` insert, `sessions` update cu starea de rutare), apoi răspunde. O eroare la W1 se loghează, iar răspunsul ajunge totuși la user.
- **Worker:** publică `{turn_id}` pe `memory.ingest` după răspuns. Coada se declară în `src/shared/rabbitmq.py`.
- **Gata când:** două request-uri consecutive cu același `session_id` produc turele 1 și 2. Testul de integrare rulează pe Postgres din `kind` / CI.

### Pasul 4: Router, cu rutare de sesiune și V0

- `src/shared/memory/tokens.py`: `count_tokens` (tokenizer-ul de referință, încărcat o dată).
- `src/shared/memory/assembly.py`: `build_messages(system, loaded_turns, candidates, prompt, budget, settings)` după 4.7, cu turele recente alese după buget și pașii de filtrare. Funcție pură: R3 se face în afara ei, printr-un callback care primește id-urile și întoarce conținutul. La V0, `candidates` e gol.
- `src/shared/memory/presets.py`: preseturile V0–V5.
- `src/router/session_routing.py`: regulile din 4.1, ca funcție pură peste `(prompt, predicții, stare sesiune)`.
- `src/router/router.py`: R1 (ultimele 20 de ture), rutarea de sesiune, `select_model`, `build_messages`. System prompt-ul conține instrucțiunile de limbă și de formatare a codului. Payload-ul spre worker conține `messages`, `turn` și `routing`.
- **Teste:** `build_messages` pe model mic și mare, cu aceeași sesiune: câte ture intră întregi, ultima tură trunchiată, sesiune care încape toată (fără fragmente), sesiune mai lungă de 20 de ture, spațiu rămas umplut cu ture mai vechi, prompt peste buget. `session_routing` pe „continua”, distribuție plată, schimbare clară de subiect, prima tură.
- **Gata când:** în cluster, „scrie o funcție de sortare în Python” urmat de „continua” rămâne pe modelul de coding, cu `inherited=true`, iar al doilea răspuns ține cont de primul.

### Pasul 5: Modulul de indexare (`src/shared/memory/`)

- `segment.py`: `Segment` și `segment(text)`, cu recunoașterea tipurilor din 4.3 (fence-uri, cod și log fără fence, LaTeX, tabele, liste, proză).
- `splitters.py`: protocolul `Splitter` și implementările `ProseSplitter`, `CodeSplitter`, `LogSplitter`, `ListSplitter`, `AtomicSplitter`, într-un dicționar `kind → splitter`.
- `chunking.py`: `chunk(text, source)`: segmentare, tăiere doar peste `MAX_TOKENS`, normalizare. `CHUNKER_VERSION` și constantele într-un singur loc. Calculează `content_hash` pe fiecare chunk. Proza userului trece prin chunking doar peste 150 de tokeni.
- `header.py`: `extract_question` (4.2) și `build_dense_query` (4.4), pe aceeași segmentare.
- `analyzer.py`: analizorul BM25.
- `embedding.py`: o interfață `EmbeddingClient`, după modelul `InferenceEngine`, cu implementarea Ollama (batch).
- **Teste:** reconstrucția textului pe răspunsuri reale, inclusiv cu fence neînchis, fence `~~~`, cod lipit fără fence în prompt, cod fără fence în răspuns și Python cu `SyntaxError`. Un bloc de cod sub `MAX_TOKENS` rămâne întreg. Fiecare splitter separat: o listă lungă se taie doar între elemente, un bloc `$$` și un tabel nu se taie, un log lung își păstrează finalul. Un răspuns mixt (proză + cod + formulă) primește tipul corect pe fiecare chunk. Un articol lipit de user (proză > 150 de tokeni) produce chunk-uri `source = 'prompt'`. Extragerea header-ului pe cele trei tipuri de ture și pe proza de peste 150 de tokeni. `build_dense_query` pe un traceback scurt (intră întreg), pe unul lung (rămân ultimii 512 tokeni, iar excepția e inclusă) și pe un prompt cu întrebare + traceback (intră doar întrebarea). Analizorul pe diacritice și identificatori.
- **Gata când:** testele trec, iar chunk-urile unor răspunsuri reale salvate arată rezonabil la inspecție manuală.

### Pasul 6: Serviciul `memory`

- `src/memory/` (`service.py`, `Dockerfile`): consumer pe `memory.ingest` cu `prefetch=1`. Citește tura, rulează modulul din pasul 5, face W2, apoi ack.
- DLQ: `x-dead-letter-exchange` pe coadă și un număr maxim de reîncercări.
- Comandă de reindexare: turele fără chunk-uri, sau cu alt `chunker_version` sau `embedding_model`.
- **Helm:** `charts/moe-router/templates/memory/deployment.yaml`, iar bge-m3 intră în lista de modele descărcate de Job-ul de setup. Setezi `keep_alive` și `OLLAMA_MAX_LOADED_MODELS` (4.4).
- **Skaffold:** un artefact nou pentru imaginea `memory`. CI: imaginea în matricea `images`.
- **Gata când:** după o conversație de 6 ture, `chunks` conține rânduri pentru toate turele, iar reprocesarea aceluiași mesaj nu dublează nimic.

### Pasul 7: Setul de date și harness-ul

- `scripts/rag_eval/`: generarea conversațiilor, fișierele de întrebări cu citate gold (JSONL), `run.py` care indexează și evaluează, și `report.py` cu tabelele.
- Harness-ul folosește direct `src/shared/memory/` și o bază Postgres locală, separată.
- **Gata când:** V0 rulează cap-coadă pe tot setul și produce tabelul gol pentru V1–V5.

### Pasul 8: V1 și V2

- `src/shared/memory/retrieval.py`: R2 (un query pe chunk-urile sesiunii, fără ultima tură, cu distanța `embedding <=> :q`), clasamentul dense și BM25 în Python, RRF, deduplicarea (4.6), top `RAG_POOL`. Coloanele se aleg după preset.
- **Teste:** deduplicarea pe același cod relipit (hash), pe codul rescris cu o modificare (near) și pe chunk-uri diferite cu termeni comuni, care trebuie să rămână amândouă. Filtrarea din `build_messages` pe candidați din ture incluse întregi.
- Rulezi V1 și V2 × top-k, plus ablațiile de header, de query și de deduplicare.
- **Gata când:** tabelul are V0–V2 complet.

### Pasul 9: V3, V4 și prompt caching (offline)

- `src/shared/memory/contextualize.py`: prompt-ul din 4.8, care completează `context`, `terms_ctx`, `embedding_ctx`, `context_model`.
- Experimentul de prompt caching cu cele trei variante.
- **Gata când:** tabelul are V3–V4 și tabelul de timp de indexare cu și fără cache.

### Pasul 10: V5 (opțional)

- Reranker-ul în spatele unei interfețe. Măsori calitatea și latența suplimentară per request.

### Pasul 11: Integrarea retrieval-ului în router

- Cea mai bună variantă rulabilă pe CPU (practic V1 sau V2) intră în router: embedding-ul query-ului și căutarea, în paralel cu rutarea de sesiune, cu fragmentele trimise în `build_messages`.
- `retrieved_chunk_ids`, numărul de ture întregi și top-k-ul efectiv merg în payload și pe tură.
- Câmpul `variant` în gateway și router, cu `ALLOW_VARIANT_OVERRIDE` (secțiunea 6).
- **Gata când:** evaluarea cap-coadă în cluster (un subset al setului de date, prin gateway) confirmă rezultatele din harness, iar latențele pe etape sunt măsurate.

---

## 8. Limitări

- **Mesaje simultane în aceeași sesiune** primesc același număr de tură, iar al doilea INSERT pică pe `UNIQUE`. Ipoteza e că clientul așteaptă răspunsul înainte de mesajul următor.
- **Un prompt ambiguu la schimbarea subiectului** rămâne pe domeniul vechi până la primul prompt clar.
- **Întrebările foarte vagi în sesiuni lungi** depind doar de turele recente (nu există rezumat).
- **Plafonul de 20 de ture întregi** se aplică și pe modelele mari. Turele mai vechi intră doar prin fragmente, chiar dacă ar încăpea.
- **Modelele mici pot omite fence-urile.** Codul nerecunoscut ajunge în chunk-uri `prose`, cu o tăiere mai puțin precisă.
- **Formulele:** căutarea semantică pe LaTeX e slabă cu un model de embedding general. Întrebările despre calcule se bazează mai mult pe proza din jurul formulelor.
- **Traducerile:** originalul și traducerea sunt în limbi diferite. bge-m3 le leagă, dar BM25 nu.
- **IDF zgomotos** în sesiunile scurte. Recall-ul se raportează în funcție de lungimea sesiunii.
- **Tokenizer de referință unic**, compensat prin marja de 10%.
- **Detectarea codului fără fence** e euristică, testată pe setul de date, nu pe cazul general.
- **Diferențe față de articol:** corpusul e o sesiune, contextul e generat de un model local mic, datele amestecă cod cu text în română, iar V3 se compară cu un baseline cu header, mai puternic decât chunk-ul gol.
ddeci