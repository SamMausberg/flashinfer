## Description

`top_p_sampling_from_probs`, `top_k_sampling_from_probs` and `top_k_top_p_sampling_from_probs(..., filter_apply_order="joint")` run one 1024-thread CTA per row. Each rejection round scans the row in 4096-element tiles until the CDF passes the uniform draw and then reads the whole row again to sum the mass above the two pivots, so a round costs up to two row reads by a single SM. At decode batch sizes most of the GPU sits idle. On GH200 with a 151936-entry vocabulary, batch 1 takes 34.6 us for top-p and 46.3 us for top-k on peaked probabilities, and 203 us for top-k on a flat distribution, which needs more rounds.

On SM90 this PR adds a split-row variant of these three samplers. Each row is divided across a thread block cluster of 2 to 8 CTAs with 256 threads each. Every CTA copies its chunk of the row into shared memory once (with `cp.async` when the vocabulary is a multiple of 4, scalar loads otherwise), and the row is not read from global memory again. A rejection round is then one pass over the shared-memory chunk that computes the mass (and, for top-k, the count) above both pivots, with per-warp partial sums written into every CTA of the cluster through distributed shared memory, followed by an inverse-CDF search that only the CTA holding the u-quantile performs (rank, then warp, then thread, then element prefix sums, all in index order). Two cluster barriers per round replace up to two row reads per round.

The algorithm is the one the single-CTA kernels use: the same Philox subsequence and number of draws, the same pivots and accept/reject rules, and the same parameter indexing per mode, including `indices` and per-request `top_k`/`top_p` tensors. Only the floating-point summation order differs, so a draw can land on the neighboring token when u falls within rounding distance of a CDF boundary. Comparing this branch with its base commit over 189,540 draws (vocab 8192 to 262144, batch 1 to 150, logits `randn * 1` and `randn * 5`, all three modes, per-row thresholds and `indices`), 438 draws (0.23%) differed; all but 3 were on the flat `randn * 1` rows, where one token's CDF step near the top of the range is only tens of float32 ulps wide. In a single-round setting (top-k with k above the vocabulary size, 28,200 draws), each of the 106 differing draws was the adjacent token. For scale, the base kernel's `deterministic=True` and `deterministic=False` variants, which differ only in the scan inside a 4096-element tile, disagreed on 9 of the same 189,540 draws.

All reductions in the split-row kernel use a fixed order, so it is deterministic whether or not `deterministic=True` is passed; for the shapes it handles, `deterministic=False` no longer selects a different kernel. That flag did not buy speed on the single-CTA path at the shapes I checked (vocab 151936: top-k batch 1 45.3 us vs 46.3 us with `deterministic=True`, joint batch 8 70.9 us vs 72.3 us).

`GetSplitRowSamplingClusterSize` picks the smallest power-of-two cluster that keeps each chunk at or below 26624 floats, so that two CTAs fit on an SM (vocabularies above 212992 get 8 CTAs with one resident per SM), and doubles it (up to 8) while the grid fills less than half of the SMs and chunks stay at 4096 floats or more. The single-CTA kernels are kept in these cases:

- Rows shorter than 8192 entries, and rows whose 8-way chunks do not fit in shared memory (above roughly 460K entries on GH200).
- Large grids that need 8-CTA clusters, where the single-CTA kernels catch up: vocabularies above 212992 at 24 or more CTAs per SM (batch 396 or more on 132 SMs), and vocabularies above 131072 at 64 or more CTAs per SM (batch 1056 or more). Measured crossovers at vocab 262144 were between batch 384 (top-p 1.03x, top-k 1.12x) and 448 (top-p 0.97x), and at vocab 151936 between batch 1024 (top-p 1.02x) and 1536 (top-p 0.98x). Clusters of 2 and 4 CTAs were at least even at every batch size I measured, up to 8192 (for example top-p at vocab 50257 and batch 8192: 2383 us before, 1220 us after).
- GPUs other than SM90, and builds where the kernel was not compiled for SM90. SM100 and newer also support clusters, but I could not measure them, so they keep the existing kernels.

`min_p_sampling_from_probs`, `sampling_from_probs`, `sampling_from_logits` and the renorm kernels are not changed. The default `filter_apply_order="top_k_first"` reaches the new kernel through `top_p_sampling_from_probs` when its top-k fast path does not apply (per-request `top_k` tensor, `indices`, k above 256, or vocabulary below 65536), and the logits entry points reach it after their softmax.

### Performance

NVIDIA GH200 480GB (sm_90a, 132 SMs), CUDA 12.8, driver 570, torch 2.11.0+cu128. Each value is the per-call time from a CUDA graph of 20 calls with distinct Philox offsets, replayed 15 times (median replay divided by 20), with the inputs resident in L2 as they are after the LM head. Probabilities are `softmax(randn(B, V) * 5)` with `torch.manual_seed(0)`; in a sample of 24 such rows at vocab 32000 to 262144, the top-1 probability ranged from 0.12 to 0.71 and a top-p 0.9 nucleus held 4 to 205 tokens. `top_k` and `top_p` are per-row tensors of 50 and 0.9, and `deterministic` is left at its default (`True`). Before is the base commit 8589d49b (the sampling code is the same on current main) and after is this branch, measured in separate processes in two interleaved rounds (base first, then branch first); the table shows the median of the two, and no cell moved by more than 1.4% between rounds.

| vocab | batch | top_p before / after (us) | top_k before / after (us) | joint before / after (us) |
|---:|---:|---|---|---|
| 8192 | 1 | 6.0 / 5.0 (1.22x) | 6.5 / 5.2 (1.25x) | 6.7 / 5.3 (1.27x) |
| 8192 | 8 | 8.0 / 6.6 (1.20x) | 7.8 / 6.2 (1.25x) | 8.8 / 6.9 (1.28x) |
| 8192 | 32 | 9.9 / 8.1 (1.23x) | 9.3 / 7.4 (1.25x) | 11.0 / 8.3 (1.32x) |
| 8192 | 128 | 11.9 / 10.4 (1.14x) | 11.9 / 10.0 (1.19x) | 13.2 / 10.9 (1.20x) |
| 8192 | 512 | 26.5 / 19.6 (1.35x) | 27.6 / 19.9 (1.39x) | 29.7 / 21.0 (1.41x) |
| 32000 | 1 | 9.5 / 6.7 (1.42x) | 10.5 / 7.0 (1.51x) | 10.8 / 7.1 (1.53x) |
| 32000 | 8 | 14.6 / 7.9 (1.84x) | 16.5 / 8.4 (1.96x) | 16.6 / 8.6 (1.94x) |
| 32000 | 32 | 18.6 / 10.7 (1.73x) | 20.7 / 11.0 (1.88x) | 21.2 / 11.4 (1.85x) |
| 32000 | 128 | 21.6 / 16.3 (1.32x) | 24.2 / 17.8 (1.36x) | 24.8 / 18.5 (1.34x) |
| 32000 | 512 | 57.4 / 40.6 (1.41x) | 64.2 / 44.6 (1.44x) | 66.0 / 45.4 (1.45x) |
| 32000 | 2048 | 182.5 / 118.3 (1.54x) | 206.4 / 132.4 (1.56x) | 210.1 / 134.8 (1.56x) |
| 50257 | 1 | 33.1 / 7.7 (4.28x) | 32.7 / 7.8 (4.21x) | 33.5 / 8.0 (4.20x) |
| 50257 | 8 | 49.4 / 9.7 (5.09x) | 52.3 / 10.3 (5.08x) | 52.9 / 10.6 (5.00x) |
| 50257 | 32 | 60.5 / 13.6 (4.44x) | 66.5 / 14.7 (4.51x) | 67.4 / 15.3 (4.41x) |
| 50257 | 128 | 74.5 / 20.0 (3.73x) | 78.3 / 23.4 (3.35x) | 79.1 / 23.5 (3.37x) |
| 50257 | 512 | 192.9 / 87.6 (2.20x) | 205.2 / 95.7 (2.14x) | 207.0 / 96.7 (2.14x) |
| 128256 | 1 | 28.6 / 8.6 (3.31x) | 33.2 / 9.4 (3.53x) | 33.1 / 9.3 (3.54x) |
| 128256 | 8 | 45.0 / 11.1 (4.05x) | 68.0 / 14.3 (4.76x) | 67.3 / 14.4 (4.69x) |
| 128256 | 32 | 56.5 / 17.8 (3.17x) | 83.4 / 21.0 (3.97x) | 83.1 / 21.5 (3.87x) |
| 128256 | 128 | 78.8 / 48.1 (1.64x) | 107.8 / 52.1 (2.07x) | 106.8 / 52.9 (2.02x) |
| 128256 | 512 | 187.0 / 147.8 (1.27x) | 235.4 / 166.3 (1.42x) | 235.9 / 168.1 (1.40x) |
| 151936 | 1 | 34.6 / 9.2 (3.77x) | 46.3 / 11.0 (4.22x) | 46.0 / 11.0 (4.18x) |
| 151936 | 8 | 52.7 / 11.1 (4.77x) | 72.9 / 13.5 (5.39x) | 72.3 / 13.6 (5.30x) |
| 151936 | 32 | 64.9 / 22.0 (2.95x) | 87.6 / 23.3 (3.76x) | 86.8 / 23.6 (3.67x) |
| 151936 | 128 | 87.1 / 59.2 (1.47x) | 114.3 / 64.6 (1.77x) | 114.0 / 65.6 (1.74x) |
| 151936 | 512 | 221.1 / 195.0 (1.13x) | 273.1 / 215.3 (1.27x) | 274.7 / 218.0 (1.26x) |
| 151936 | 1024 | 386.4 / 377.5 (1.02x) | 480.7 / 421.0 (1.14x) | 484.2 / 425.3 (1.14x) |
| 151936 | 2048 | 724.5 / 724.5 (single-CTA kernel kept) | 890.2 / 890.6 | 899.6 / 900.2 |
| 201088 | 1 | 43.4 / 10.3 (4.23x) | 56.0 / 12.2 (4.58x) | 55.6 / 12.2 (4.54x) |
| 201088 | 8 | 67.5 / 12.8 (5.28x) | 86.4 / 15.3 (5.64x) | 85.5 / 15.4 (5.54x) |
| 201088 | 32 | 83.1 / 24.6 (3.38x) | 116.2 / 27.4 (4.24x) | 115.4 / 27.9 (4.13x) |
| 201088 | 128 | 119.1 / 67.2 (1.77x) | 169.6 / 75.7 (2.24x) | 170.1 / 76.7 (2.22x) |
| 201088 | 512 | 290.8 / 227.4 (1.28x) | 367.6 / 254.7 (1.44x) | 369.3 / 257.3 (1.44x) |
| 262144 | 1 | 61.2 / 12.0 (5.11x) | 84.2 / 14.7 (5.75x) | 83.4 / 14.8 (5.65x) |
| 262144 | 8 | 85.0 / 14.6 (5.81x) | 127.1 / 19.6 (6.48x) | 125.8 / 20.0 (6.28x) |
| 262144 | 32 | 112.2 / 34.6 (3.24x) | 150.4 / 40.6 (3.71x) | 149.1 / 40.9 (3.65x) |
| 262144 | 128 | 163.8 / 107.2 (1.53x) | 216.6 / 123.4 (1.75x) | 215.8 / 124.7 (1.73x) |
| 262144 | 256 | 244.4 / 199.8 (1.22x) | 314.5 / 233.9 (1.34x) | 314.7 / 235.2 (1.34x) |
| 262144 | 512 | 372.1 / 371.7 (single-CTA kernel kept) | 474.5 / 473.9 | 475.3 / 475.3 |

The gain is largest at batch 1 to 32, where the old kernel leaves most SMs idle, and shrinks as the grid fills the GPU; the smallest gain on the new path is 1.02x (top-p, vocab 151936, batch 1024). Vocabulary 50257 gains more than its size suggests because an odd row length forces the single-CTA kernel onto scalar loads in every pass, while the split-row kernel pays for scalar loads once.

Flat distributions need more rejection rounds, and the gain grows with the round count. Same setup with `softmax(randn(B, V))`, where a top-p 0.9 nucleus holds about 61% of the vocabulary:

| vocab | batch | top_p before / after (us) | top_k before / after (us) | joint before / after (us) |
|---:|---:|---|---|---|
| 32000 | 1 | 10.3 / 6.3 (1.63x) | 35.1 / 15.8 (2.22x) | 35.0 / 16.0 (2.19x) |
| 32000 | 32 | 17.7 / 10.2 (1.74x) | 74.6 / 31.8 (2.35x) | 74.3 / 32.3 (2.30x) |
| 151936 | 1 | 32.9 / 8.8 (3.76x) | 203.1 / 32.5 (6.25x) | 201.2 / 33.4 (6.02x) |
| 151936 | 32 | 69.2 / 22.2 (3.12x) | 342.4 / 63.9 (5.36x) | 337.8 / 65.9 (5.13x) |
| 262144 | 1 | 53.3 / 11.0 (4.84x) | 365.1 / 44.5 (8.20x) | 361.1 / 46.6 (7.75x) |
| 262144 | 32 | 109.3 / 33.2 (3.29x) | 611.6 / 116.0 (5.27x) | 604.6 / 121.9 (4.96x) |

To reproduce the tables, run this script on each revision, for example `python bench.py 151936:1:5:top_p 151936:8:5:joint` (arguments `vocab:batch:logit_std:mode`); on the shapes I rechecked it gives the table values to within 0.1 us.

<details>
<summary>bench.py</summary>

```python
import statistics
import sys

import torch

import flashinfer

# usage: python bench.py vocab:batch:std:mode ...   (mode = top_p | top_k | joint)
for case in sys.argv[1:]:
    v, b, std, mode = case.split(":")
    v, b, std = int(v), int(b), float(std)
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(b, v, device="cuda") * std, dim=-1)
    k = torch.full((b,), 50, dtype=torch.int32, device="cuda")
    p = torch.full((b,), 0.9, device="cuda")
    fn = {
        "top_p": lambda o: flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, seed=1, offset=o
        ),
        "top_k": lambda o: flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, seed=1, offset=o
        ),
        "joint": lambda o: flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="joint", seed=1, offset=o
        ),
    }[mode]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(20):  # 20 calls with distinct Philox offsets
            fn(i * 64)
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 20)
    print(f"{v},{b},{std},{mode},{statistics.median(times):.1f} us")
```

</details>

The existing benchmark routines show the same trend. They use cold L2, uniform random probabilities (a flat distribution), scalar `top_k=50` and `top_p=0.9`, and a fixed seed and offset, and they print milliseconds with three decimals:

```bash
for r in top_p_sampling_from_probs top_k_sampling_from_probs top_k_top_p_sampling_from_probs; do
  for vb in "151936 1" "151936 32" "151936 512" "262144 8"; do
    set -- $vb
    python benchmarks/flashinfer_benchmark.py --routine $r --filter_apply_order joint \
      --batch_size $2 --vocab_size $1 --top_k 50 --top_p 0.9
  done
done
```

| vocab | batch | top_p before / after (us) | top_k before / after (us) | joint before / after (us) |
|---:|---:|---|---|---|
| 151936 | 1 | 31 / 8 | 238 / 38 | 234 / 40 |
| 151936 | 32 | 80 / 22 | 616 / 96 | 607 / 100 |
| 151936 | 512 | 230 / 195 | 1653 / 859 | 1639 / 877 |
| 262144 | 8 | 72 / 16 | 934 / 97 | 924 / 103 |

## Related Issues

None.

## Pull Request Checklist

### Pre-commit Checks

- [x] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [x] I have installed the hooks with `pre-commit install`.
- [x] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

## Tests

- [x] Tests have been added or updated as needed.
- [x] All tests are passing (`unittest`, etc.).

On the GH200 (sm_90a), JIT build for 9.0a:

- `pytest tests/utils/test_sampling.py -k "sampling and not softmax and not blackwell and not renorm and not mask and not chain and not speculative and not freq"`: 399 passed, 9 skipped.
- `pytest tests/utils/test_sampling.py -k "freq and not softmax and not logits"`: 87 passed, 3 skipped. This includes the new frequency test and the existing 5,000,000-draw frequency tests, which at vocab 32000 and 128256 take the new path with clusters of 2 and 8.

Two tests are added. `test_split_row_rejection_sampling` checks every sample against the top-k/top-p masks for shapes that use clusters of 2, 4 and 8, an odd vocabulary (50257, scalar staging), per-row `top_k`/`top_p`, `k = 1`, a row whose first half has zero probability, `indices`, and repeatability for a fixed seed and offset. `test_split_row_rejection_sampling_freq` puts the probability mass on a sparse set that covers both sides of every chunk and warp-segment boundary for clusters of 2, 4 and 8, draws 51,200 samples per case at decode batch sizes, and requires every token frequency to be within 6 standard deviations of the renormalized probability. To check that the frequency test detects a subtle error, I removed the warp-level prefix subtraction from the kernel; all 9 frequency cases failed while the mask-based test still passed.

`compute-sanitizer` (2025.1) `memcheck`, `racecheck` and `synccheck` report no errors or hazards on the split-row kernel for vocab 8192, 32000, 50257, 151936 and 262144 in all three modes, including an all-zero row. Only sm_90a was tested.

## Reviewer Notes

- The split-row kernel ignores `deterministic=False`, because its reductions are already fixed-order. The Description gives the old path's timing with that flag.
- `top_k_sampling_from_probs` reads `top_k_arr` by output index and the joint sampler by row index, as the existing kernels do. Open PRs #5340 and #5324 change that indexing in the single-CTA kernels; whichever lands second needs the same change in `SplitRowRejectionSamplingFromProbKernel`. #5352 changes `DeviceSamplingFromProb` in the single-CTA kernels, which the new kernel does not use; it still applies to the shapes that keep the single-CTA path, and the "before" numbers here do not include it.
- The launch path adds three `cudaDeviceGetAttribute` queries and a `cudaFuncGetAttributes` call per launch, in the style of the existing launchers. In eager mode at batch 1 (vocab 151936, top-p) a call went from 29.9 us to 14.4 us of wall time, which is host-bound after this change.
- The tuning constants (26624-float target chunk, 4096-float minimum chunk, the two large-grid cutoffs) come from GH200 measurements and are expressed per SM count; other SM90 parts (H100 PCIe, H20) were not measured.

Developed in combination with Claude Code.
