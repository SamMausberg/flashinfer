# FlashInfer PR drafts

Finished, reviewed FlashInfer changes that have not been opened as upstream PRs yet. Each one has a branch on
this fork and a folder here:

- `drafts/<name>/title`: the PR title
- `drafts/<name>/body.md`: the PR description, ready to paste
- `drafts/<name>/branch`: the branch on SamMausberg/flashinfer
- `drafts/<name>/notes.md`: what was validated, on what, and anything to do before submitting
- `drafts/<name>/*.csv`: raw benchmark data behind the tables in the description

All of them were developed and validated on an NVIDIA GH200 (sm_90a, CUDA 12.8, torch 2.11.0+cu128).

## Submitting one

1. Bring the branch up to date. Upstream moves quickly, so rebase first:

   ```bash
   git clone https://github.com/SamMausberg/flashinfer.git && cd flashinfer
   git remote add upstream https://github.com/flashinfer-ai/flashinfer.git
   git fetch upstream && git checkout <branch>
   git rebase upstream/main
   git push --force-with-lease origin <branch>
   ```

   If the rebase touched the same files, rerun the tests listed in `notes.md` on a GPU before pushing.
2. Check that nobody opened the same change in the meantime:
   `gh pr list --repo flashinfer-ai/flashinfer --state open --search "<keywords>"`.
3. Open it: `./submit.sh <name>` from a checkout of this `pr-drafts` branch, or open the branch on GitHub, click
   "Compare & pull request" against `flashinfer-ai/flashinfer:main`, and paste `title` and `body.md`.

After opening, pre-commit, Build Docs, and the documentation checks run on their own. "Test Results Summary"
fails until a maintainer authorizes GPU CI; that is expected.

## Drafts

See `./submit.sh --list` for the current set, and each `notes.md` for per-PR details.
