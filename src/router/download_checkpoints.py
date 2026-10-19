"""
download_checkpoints.py -- the router pod's init container: fetch the private Laya classifiers.

The router image ships WITHOUT weights (the image is public on GHCR, the classifiers are
private on Hugging Face). Before the router starts, this script downloads both checkpoints
into CHECKPOINT_DIR (a volume shared with the router container):

    CHECKPOINT_DIR/laya-domain-v4/
    CHECKPOINT_DIR/laya-complexity-v2/

Settings (shared/config.py, CheckpointSettings): HF_TOKEN (required), CHECKPOINT_DIR,
DOMAIN_REPO / DOMAIN_REVISION, COMPLEXITY_REPO / COMPLEXITY_REVISION.

On a persistent volume, a restart only re-checks the files: huggingface_hub keeps its download
metadata in <dir>/.cache and skips files that haven't changed.

Locally:  HF_TOKEN=hf_... CHECKPOINT_DIR=src/router/checkpoints python src/router/download_checkpoints.py
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ (or /app in the image)
from shared.config import get_checkpoint_settings  # noqa: E402

# folder names the router expects (DOMAIN_CHECKPOINT / COMPLEXITY_CHECKPOINT point here)
DOMAIN_DIR = "laya-domain-v4"
COMPLEXITY_DIR = "laya-complexity-v2"


def main() -> None:
    cfg = get_checkpoint_settings()
    out = Path(cfg.checkpoint_dir)
    for repo, revision, name in [
        (cfg.domain_repo, cfg.domain_revision, DOMAIN_DIR),
        (cfg.complexity_repo, cfg.complexity_revision, COMPLEXITY_DIR),
    ]:
        print(f"[*] {repo}@{revision} -> {out / name}")
        path = Path(snapshot_download(
            repo_id=repo, revision=revision, local_dir=str(out / name),
            token=cfg.hf_token.get_secret_value(),
            ignore_patterns=["*.md", ".gitattributes", "optimizer*", "training_args*"],
        ))
        print(f"[OK] {sorted(p.name for p in path.iterdir() if not p.name.startswith('.'))}")


if __name__ == "__main__":
    main()
