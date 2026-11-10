# scripts

Tools run by hand or by the setup Job; nothing here runs inside a service.

| path | what | when |
|---|---|---|
| `seed_models.py` | load a catalog YAML into the `models` table (validated with the router's rule) | setup Job; locally after editing the catalog |
| `manage_models/` | `manage_models.py --provision / --prune / --delete`: pull or remove Ollama models of the catalog; `wait_for.py`: wait for TCP services. Its own image (`tools`) | setup Job |
| `laya/` | building the classifier datasets, training and evaluating Laya (git-ignored, local only) | retraining the classifiers |
| `rag_eval/` | RAG evaluation: dataset, offline harness, cluster run, reports. See `rag_eval/README.md` | after changing chunking / retrieval / assembly |
| `classifier_eval/long_prompts.py` | the classifiers on long prompts (document + request at the start / end), input strategies x windows | after changing `classifier_input.py` or a checkpoint |
| `resources/` | raw prompt datasets (git-ignored) | |

Results go to `*/results/<timestamp>/` (git-ignored).

```bash
python scripts/classifier_eval/long_prompts.py --max-lens 512 1024 --strategies raw edges --sizes 200 700 1500
```
