# CPU inference study: batching and BatchNorm folding

## Question

Can a fixed trained character-level MLP execute more efficiently through batching and evaluation-time BatchNorm folding while retaining numerically close logits?

## Model and run

The supplied notebook defines a 27-character vocabulary, three-character context, embedding width 10, five hidden layers of width 100 with Tanh, and 27 output logits. There are six Linear–BatchNorm pairs and five Tanh layers: 17 layer objects originally and 11 after folding. The embedding table is unchanged. Parameter count before folding is 47,551.

The saved training configuration is 10,000 steps, batch size 32 and learning rate 0.1. It uses custom sample-variance BatchNorm with momentum 0.5 and epsilon 1e-5. Dataset partitioning is 80/10/10 by word, with two seeded shuffles in a full run. Saved context counts are 182,580 training, 22,767 validation and 22,799 test. No test-set quality claim is made.

The author reported running in Colab. CPU assertions passed; two logical CPUs were visible. All reported inference benchmarks request one PyTorch computation thread. CPU model, original software versions, input-data checksum and trained checkpoint were not captured in the uploaded notebook. The code normally creates floating-point tensors with the runtime default, but the historical dtype was not explicitly logged.

## Transformation

For an evaluation-mode linear output z = xW + b, BatchNorm computes:

    y = gamma * (z - mean) / sqrt(variance + epsilon) + beta
    scale = gamma / sqrt(variance + epsilon)
    W_folded = W * scale
    b_folded = (b - mean) * scale + beta
    y = x @ W_folded + b_folded

Weights use [input_features, output_features] layout, so scale multiplies columns. Fixed running statistics make this rearrangement possible. Training-mode BatchNorm depends on the current batch and is not replaced by this fixed transformation. Tanh remains between folded layers. Rebuild the folded model after changes to trained parameters or running statistics.

## Measurement method

Every workload processes the first 128 validation contexts with batch sizes 1, 8, 32 or 128. All contexts are already available: this is not an online request-arrival experiment. Both models use the same wrapper, input slicing and Python loop, under inference_mode. Timed work includes model execution and wrapper overhead; it excludes folding construction, training, output comparisons, printing, softmax and character sampling.

The folding comparison uses torch.utils.benchmark.Timer, num_threads=1 and blocked_autorange(min_run_time=2.0). Round 1 measures original then folded at each batch size; round 2 reverses model order. Batch-size order remains ascending. Results are medians; IQR/median quantifies spread and is not a confidence interval. Individual timing samples were not exported. Two rounds do not establish statistical significance or independence from environmental conditions.

## Correctness

First, output agreement between individual and batched inference passed on 128 contexts. After folding, original and folded logits passed rtol=1e-4 and atol=1e-5 checks at all four batch sizes on those contexts.

A broader comparison then processed the complete validation split in batches of 128:

| Metric | Recorded value |
|---|---:|
| Contexts | 22,767 |
| Original mean cross-entropy | 2.2491625373100748 |
| Folded mean cross-entropy | 2.2491625567463154 |
| Absolute loss difference | about 1.94e-8 |
| Maximum absolute logit difference | 2.86102294921875e-6 |
| Changed argmax predictions | 0 |

The mean is computed from summed per-example losses, divided by the total count. This handles the last partial batch. These checks establish agreement on tested inputs, not universal equivalence in finite precision or perfect prediction accuracy. The loss difference is consistent with rounding after arithmetic rearrangement.

## Folding results

All times below refer to **128 contexts**, not one input or one name. All observations, including noisy ones, are retained.

| Round | Batch | Original ms | Folded ms | Reported speedup | Original IQR/median | Folded IQR/median |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 78.866 | 48.037 | 1.64x | 104.9% | 72.1% |
| 1 | 8 | 11.664 | 2.712 | 4.30x | 100.4% | 12.3% |
| 1 | 32 | 1.317 | 0.703 | 1.87x | 4.6% | 2.4% |
| 1 | 128 | 0.725 | 0.556 | 1.30x | 2.4% | 44.0% |
| 2 | 1 | 23.470 | 12.659 | 1.85x | 2.1% | 11.7% |
| 2 | 8 | 3.528 | 1.456 | 2.42x | 2.6% | 1.5% |
| 2 | 32 | 1.311 | 0.702 | 1.87x | 2.9% | 2.4% |
| 2 | 128 | 0.735 | 0.738 | 1.00x | 37.1% | 5.9% |

The strongest repeatable observation is batch size 32: 1.317 to 0.703 ms and 1.311 to 0.702 ms, approximately 1.87x in both rounds, with relatively low reported variability. Round 2 implies about 46.5% lower workload time. A 1.87x speedup is not an 87% time reduction.

Batch sizes 1 and 8 favour folding in both rounds but have severe round-1 variability. The 4.30x observation should not be the headline. At batch 128, one round favours folding and the other shows virtually equal medians (rounded speedup 1.00x; folded is slightly slower). The evidence does not establish a consistent advantage there.

## Batching findings and interpretation

The saved initial batching run reports 26.366, 3.743, 1.390 and 0.705 ms respectively. Additional rounds and IQR diagnostics show variation; their complete printed results are retained in results/saved_outputs.txt. They are separate measurements and should not be combined with folding measurements to claim a compounded speedup.

Folding removes 768, 96, 24 or 6 BatchNorm calls per 128-context workload at batch sizes 1, 8, 32 or 128. It removes both Python calls and normalization tensor operations. Fewer calls and less intermediate work are plausible mechanisms, but the study does not profile their separate contributions. Matrix-multiplication count remains unchanged.

## Limits and next steps

This is one small model, one author-reported environment, a short training run and a fixed workload reused during timing. Cached workloads need not reflect broader serving traffic. No memory, energy, request latency, other-hardware or optimized-library baseline measurements were made. Thread-count experiments were discussed but no such results are present in this artifact. BatchNorm folding is an established transformation; novelty is not claimed.

The original tensors are not in the notebook file. The missing checkpoint, data checksum and software versions limit exact reproducibility. For future runs, record environment metadata, data hash, checkpoint and raw timing samples before experimentation. Preserve this run as historical evidence rather than assigning later environment information to it.

## Attribution

Learning foundation: Andrej Karpathy's makemore (https://github.com/karpathy/makemore); precise source/version should be confirmed by the author. ChatGPT assisted in the experiment design, folding implementation, explanations and documentation. The recorded outputs were supplied by the author; packaging did not independently reproduce training or speedups.
