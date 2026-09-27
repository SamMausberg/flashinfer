# split-row-sampling

- Branch: `perf/split-row-sampling` on SamMausberg/flashinfer, head `e5d70344`, one commit on upstream main `05ebb2d7`.
- Changes `include/flashinfer/sampling.cuh` (+411) and `tests/utils/test_sampling.py` (+136).
- Validated on NVIDIA GH200 (sm_90a), CUDA 12.8, torch 2.11.0+cu128. The new path is enabled on SM90 only; other
  architectures keep the existing kernels.
- Reviewed independently: compute-sanitizer memcheck, racecheck and synccheck clean; `tests/utils/test_sampling.py`
  sampling subset 399 passed / 9 skipped, frequency tests 87 passed / 3 skipped; pre-commit on all files and the
  public API / doc checkers pass.
- Timing data behind the PR tables: `review-timings.csv` (reviewer's re-measurement, used in the body) and
  `author-sweep.csv` (author's original sweep, noisier baseline).

## Before submitting

- Rebase on upstream main and rerun `pytest tests/utils/test_sampling.py` on an SM90 GPU if sampling files changed.
- Open PRs #5340 and #5324 change how the existing kernels index per-request `top_k` (and `min_p`) when `indices`
  is set. If either has merged, apply the same indexing to the new split-row kernels before submitting.
- Behavior note already stated in the body: the new kernels give the same result regardless of the
  `deterministic` flag.
- Overlap check at the time of writing: no open PR does the same thing (#5352, #1561, #5607, cake_sampling are
  separate work).
