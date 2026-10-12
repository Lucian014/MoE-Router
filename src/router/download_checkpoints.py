"""
download_checkpoints.py -- used ONLY at image build time (src/router/Dockerfile, stage `checkpoints`).

Downloads the fine-tuned Laya classifiers from the private Hugging Face repos into
/checkpoints/<name>. The token comes from a build secret mounted at /run/secrets/hf_token
(podman/docker build --secret id=hf_token,env=HF_TOKEN) and is never stored in the image.

Env (set as ARG in the Dockerfile):
  DOMAIN_REPO, DOMAIN_REVISION, COMPLEXITY_REPO, COMPLEXITY_REVISION
"""
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

SECRET = Path("/run/secrets/hf_token")
OUT = Path("/checkpoints")

token = SECRET.read_text().strip() if SECRET.exists() else ""
if not token:
    sys.exit("[!] no HF token: build with  --secret id=hf_token,env=HF_TOKEN  "
             "(and `export HF_TOKEN=hf_...` in the same terminal)")

for repo, rev, name in [
    (os.environ["DOMAIN_REPO"], os.environ["DOMAIN_REVISION"], "laya-domain-v4"),
    (os.environ["COMPLEXITY_REPO"], os.environ["COMPLEXITY_REVISION"], "laya-complexity-v2"),
]:
    path = Path(snapshot_download(
        repo_id=repo, revision=rev, local_dir=str(OUT / name), token=token,
        ignore_patterns=["*.md", ".gitattributes", "optimizer*", "training_args*"],
    ))
    shutil.rmtree(path / ".cache", ignore_errors=True)   # download metadata, not needed
    print(f"[OK] {repo}@{rev} -> {path}: {sorted(p.name for p in path.iterdir())}")