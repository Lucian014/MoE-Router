# Planuri de viitor

Notat pe 2026-11-19. Ordinea propusă: ingress → observabilitate → autentificare → interfață; cererile în paralel (§5) după observabilitate, ca efectul să se vadă în grafice; mutarea pe altă infrastructură (§6) la final, după interfață. Interfața are nevoie de autentificare (cont, conversațiile mele) și de metrici (graficele), deci vine ultima. Detaliile RAG rămase deschise sunt în `rag-planv2.md`, secțiunile 8–9.

## 1. Ingress (fără port-forward)

- Clusterul are deja Traefik (implicit în k3s), expus pe 80/443 pe toate nodurile (`192.168.56.101–104`); nodul `k8s-route` are rolul `ingress`.
- De făcut: `charts/moe-router/templates/gateway/ingress.yaml` + `gateway.ingress.{enabled, host, className: traefik}` în values, activat în `values-virtualbox.yaml`; pe stație `192.168.56.103  llm-router` în `/etc/hosts`.
- De verificat: un răspuns lung (până la `rpcTimeoutSeconds: 300`) trece prin Traefik fără timeout.
- Gateway-ul nu are autentificare: ok cât timp e doar pe rețeaua host-only. Înainte de orice expunere mai largă → secțiunea 3 (sau, provizoriu, un middleware `basicAuth` în Traefik).

## 2. Observabilitate: Prometheus + Grafana

- `kube-prometheus-stack` (Prometheus, Grafana, node-exporter, kube-state-metrics) într-un namespace separat (`monitoring`); ~1–2 GB RAM, încape pe `k8s-infra`.
- Fiecare serviciu expune `/metrics` (`prometheus_client`) + un `ServiceMonitor` în chart (opțional, `metrics.enabled`):

| Serviciu | Metrici |
|---|---|
| gateway | cereri pe status, latența cap-coadă, timeout-uri RPC |
| router | decizii pe domeniu / complexitate / model / motiv (`match`, `fallback`…), timpul clasificatorului, retrieval (candidați, fragmente, tokeni de context, ms pe embed / search / rank), erori de retrieval |
| worker | latență, tokeni de intrare / ieșire, tokeni/s, răspunsuri trunchiate, pe model |
| memory | ture indexate, întârzierea indexării, mesaje în DLQ |
| RabbitMQ / Postgres | exporterele lor (lungimea cozilor, conexiuni) |

- Grafana: un dashboard „routing” (ce modele primesc ce), unul „memorie / RAG” (fragmente per cerere, cât de des pragul lasă contextul gol), unul „infrastructură”.
- Atenție la cardinalitate: fără `session_id` / `user_id` ca label.

## 3. Conturi, autentificare și autorizare

**Două lucruri diferite, ambele numite „sesiune”:**

- **Autentificarea** spune *cine* trimite cererea: un cont (`users`), un token de acces JWT scurt (~15 min) și un refresh token (cookie `httpOnly`, rotit, revocabil într-o tabelă). Din JWT rezultă doar `user_id`.
- **Conversația** (tabela `sessions` de acum, cu turele ei) e un fir de discuție. Un utilizator are oricâte. Primește o coloană `user_id` (proprietarul). În interfață și în documentație o numim „conversație”, ca să nu se confunde cu sesiunea de login.

**Cum se leagă:** gateway-ul (singurul punct public) validează JWT-ul → `user_id`; la `session_id` existent verifică `sessions.user_id == user_id`, altfel 404 (nu 403, ca să nu confirme că există); la conversație nouă o creează cu proprietarul. Router-ul și worker-ul nu știu de JWT: primesc `user_id` în mesaj doar ca informație (și pentru metrici agregate).

**Reguli care închid accesul la conversațiile altora:**

- `user_id` vine *numai* din JWT-ul validat, niciodată din corpul cererii; gateway-ul îl pune în mesajul către router / worker.
- `session_id` primit de la client trebuie să existe și să fie al lui; o conversație nouă primește id-ul de la gateway (clientul nu trimite `session_id`), iar worker-ul o creează cu `ensure(session_id, user_id)` (coloana `sessions.user_id` există deja, nefolosită: devine FK spre `users.id`, `NOT NULL` după migrare).
- Orice citire pentru utilizator filtrează în SQL: `WHERE id = :sid AND user_id = :uid` (un singur helper `get_owned_session`), nu „citește, apoi compară” în mai multe locuri.
- Teste: A nu vede / nu continuă conversația lui B (404), userul normal pe `/admin/*` → 403, token expirat → 401.

**Admin vs user:** coloana `users.role` (`user` | `admin`), pusă în JWT; endpoint-urile `/admin/*` cer `admin` (dependency `require_admin`, care recitește rolul din DB la acțiuni de scriere). Adminul schimbă catalogul din `model_registry` (enabled, default, tier, categorie, context), iar router-ul preia singur schimbarea la refresh-ul catalogului. Un model *nou* are nevoie și de worker deployat (chart) — din interfață se pot alege doar modelele deja deployate. Primul admin: creat de jobul de setup din Secret; `register` creează mereu `user`.

**De decis:**

- Implementare proprie (FastAPI + `argon2` pentru parole + `PyJWT`) sau `fastapi-users`; un IdP extern (Keycloak / Authentik, OIDC) e prea mult pentru început.
- Unde e secretul JWT: Secret în chart, ca parola Postgres.
- Roluri: măcar `user` și `admin` (admin: catalogul de modele, toate conversațiile, metricile).
- Gateway-ul are nevoie de acces la Postgres (conturi, lista conversațiilor) — acum vorbește doar cu RabbitMQ.
- Migrare: conversațiile existente rămân fără proprietar (sau ale unui cont `admin`).
- Clienții de test (`rag_eval`, curl) primesc un token de serviciu sau un cont de test.

## 4. Interfața (făcută de mine, simplă)

- Servită de gateway ca fișiere statice pe `/` (același origin: fără CORS, un singur ingress); HTML/JS simplu sau React + Vite, fără server separat.
- **Pagini:** login / cont nou; conversații (listă + chat); pentru admin: catalogul de modele și Grafana.
- **Ce vede userul normal, la fiecare răspuns:** clasificarea (domeniul) și complexitatea, modelul care a răspuns, dacă decizia a fost moștenită din tura anterioară (`inherited`). Vin din răspunsul la `/generate` (`routing`) și, pentru istoric, din `turns` (`predicted_domain`, `predicted_complexity`, `effective_domain`, `effective_complexity`, `inherited`, `model_id` — deja salvate de worker). Fără Prometheus și fără grafice pentru user. Detaliile de RAG (fragmente, tokeni de context) rămân doar pentru admin.
- **Ce vede adminul:** Grafana „ca la carte” — dashboard-urile complete din secțiunea 2, servite sub `llm-router/grafana` (același domeniu: `GF_SERVER_ROOT_URL`, `GF_SERVER_SERVE_FROM_SUB_PATH=true`), ca link din interfață sau încorporate (`GF_SECURITY_ALLOW_EMBEDDING=true`, `iframe` cu `d-solo/...`).
- **Protecția Grafana fără al doilea login:** Grafana cu acces anonim `Viewer`, dar în spatele unui middleware Traefik `forwardAuth` care întreabă gateway-ul (`GET /auth/verify?role=admin`) dacă JWT-ul din cookie e al unui admin; altfel 401/403 și cererea nu ajunge la Grafana. (Alternativă: `auth.jwt` nativ în Grafana, cu tokenuri semnate RS256.)
- **Endpoint-uri noi în gateway:** `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`; `GET /conversations`, `GET /conversations/{id}` (turele), textele fragmentelor după id; eventual streaming la `/generate`.
- Atașamentele separate de întrebare se fac odată cu interfața: `rag-planv2.md` §9.1.

## 5. Cereri în paralel

Starea de acum (2026-11-19), pe componente:

| Componentă | Acum | Ce limitează |
|---|---|---|
| gateway | async, RPC cu `correlation_id` → deja multe cereri simultane | nimic important |
| router | `prefetch_count=1` fix în cod (`router.py`), un singur pod | o cerere la un moment dat; clasificatorul rulează în `asyncio.to_thread`, deci un prefetch mai mare ar lucra deja în paralel, dar PyTorch pe CPU concurează pentru aceleași nuclee; `replicas > 1` e blocat de PVC-ul de checkpoint `ReadWriteOnce` |
| worker | `WORKER_PREFETCH=1` (configurabil per model în chart), 1 pod per model | o generare per model la un moment dat |
| Ollama | 1 pod, CPU (`gpu: 0`), fără `OLLAMA_NUM_PARALLEL` | pe CPU, generările paralele își împart nucleele: throughput-ul total crește puțin, latența fiecăreia crește; fiecare slot paralel ține propriul context în memorie |
| memory | `prefetch_count=1` | indexarea e asincronă, deci întârzie doar retrieval-ul turelor foarte recente |

**De făcut:**

- `ROUTER_PREFETCH` (config, nu constantă) + limită de thread-uri PyTorch (`torch.set_num_threads`) ca să nu se calce clasificările între ele; mai multe replici de router: checkpoint-urile în imagine sau PVC `ReadOnlyMany` / descărcate în `emptyDir` de init container.
- Workeri: `prefetch` > 1 doar împreună cu `OLLAMA_NUM_PARALLEL` (altfel cererile stau la coada Ollama, nu în RabbitMQ); scalare pe lungimea cozii (KEDA cu scaler RabbitMQ) pentru modelele folosite des.
- `MEMORY_PREFETCH` configurabil (embedding-urile bge-m3 se pot face în paralel).
- **Aceeași conversație, două cereri simultane:** acum ambele calculează același număr de tură, iar a doua nu se salvează (`TurnAlreadyExistsError` în worker). De ales: gateway-ul refuză (409) o cerere nouă cât timp conversația are una în lucru (interfața oricum blochează butonul), sau numărul turei se alocă atomic (advisory lock pe `session_id`, ca la numerotarea facturilor). Cereri paralele în conversații *diferite* nu au problema asta.
- Test de încărcare: N utilizatori simultani (ex. `locust` sau un script `asyncio` + `httpx`) → latența p50/p95 și throughput-ul în Grafana, înainte și după schimbări. Limita reală e hardware-ul (CPU pentru Ollama); GPU-ul schimbă complet calculul.

## 6. Mutarea pe altă infrastructură (după interfață)

**Ținta:** gateway + router (+ Traefik) pe VM-uri separate; Postgres, RabbitMQ și memory pe un nod de infra separat; generarea pe mașina cu A100, prin VPN.

Ce ajută: chart-ul e deja împărțit pe noduri prin `nodeSelector` într-un fișier de values (`values/values-virtualbox.yaml`); pentru noua infrastructură = un `values-<nume>.yaml` nou, cu alte hostname-uri. Catalogul suportă deja motoare externe: `engine: vllm` + `inference_host` (adresa serverului vLLM al modelului).

**Nodul A100 — de lămurit înainte de orice:**

- Mașina e partajată, cu reguli: fără sudo, fără instalări globale, porturi doar pe `127.0.0.1`, serverele oprite când nu sunt folosite. Ca nod Kubernetes (agent k3s, driver/device plugin NVIDIA) ar cere root → posibil doar cu acordul administratorilor.
- **Varianta compatibilă cu regulile (recomandată):** A100 nu intră în cluster; vLLM (și eventual Ollama pentru embedding-uri) rulează ca procese ale utilizatorului, ca la evaluări; workerii din cluster le apelează prin `inference_host`. Legătura: un tunel SSH (`autossh -L`) dintr-un pod / VM al clusterului prin VPN, pentru că porturile sunt doar pe localhost.
- Ce schimbă: un proces vLLM servește un singur model → modelele din catalog grupate pe câteva servere (`--gpu-memory-utilization` împărțit), sau un model mare per categorie în loc de multe mici; Ollama pe GPU pentru restul. Catalogul se reface pentru A100 (modele mai mari, `context_window` 32k: aici retrieval-ul ajunge la ~98% gold în context).
- Riscuri: disponibilitatea (mașină partajată, „oprit când nu e folosit” vs. un serviciu 24/7), latența și stabilitatea VPN-ului (timeout-uri + reconectare în worker / tunel), cine pornește serverele după un restart.

**Pași:**

1. VM-urile noi + k3s (control plane, nod gateway/router, nod infra); Traefik + ingress (§1), DNS / `hosts`, TLS (cert-manager) dacă iese din rețeaua locală.
2. Înainte: `runAsUser: 999` pe Postgres / RabbitMQ (§7), ca un kernel nou să nu le mai blocheze.
3. Datele: `pg_dump -Fc` pe vechiul Postgres → `pg_restore` pe cel nou (conturi, conversații, ture, chunk-uri cu embedding-uri, catalog); RabbitMQ nu are date de mutat (cozile se recreează); checkpoint-urile router-ului se descarcă din nou.
4. A100: serverele vLLM + tunelul, catalogul nou cu `engine: vllm`, testat întâi cu un singur model.
5. Prometheus / Grafana (§2) mutate odată cu clusterul; test de încărcare (§5) pe noua infrastructură.
6. Rulare în paralel vechi / nou până la confirmare, apoi oprirea clusterului VirtualBox.

## 7. Infrastructură

- `k8s-infra` e fixat pe kernelul 7.0.0-38 (GRUB, după id): pe 7.0.0-42 `gosu` se blochează, iar Postgres și RabbitMQ nu mai pornesc. Soluția durabilă: `securityContext.runAsUser: 999` pe Postgres și RabbitMQ în chart (entrypoint-ul nu mai apelează `gosu`), apoi revenirea la kernelul curent. Și: `Automatic-Reboot "false"` în `unattended-upgrades` pe noduri.
