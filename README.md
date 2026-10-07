# MLP inference: batching and BatchNorm folding

A small CPU inference study by Mushaf Ali Mir, completed October 2026.

**Main result:** at batch size 32, folding six Linear–BatchNorm pairs produced approximately **1.87x speedup in both measurement rounds** for a fixed workload of 128 contexts. All 22,767 validation contexts passed numerical comparison, with no changed highest-scoring predictions.

This is an educational implementation and performance investigation of a known technique. Results apply to this workload and recorded environment; they are not a universal acceleration claim.

## Start here

- [Annotated notebook with saved outputs](mlp_inference_study.ipynb)
- [Technical report](REPORT.md): equations, setup, all folding results, and limitations
- [Machine-readable results](results/results.json)
- [Original printed output transcript](results/saved_outputs.txt)
- [Publishing guide and draft announcement](PUBLISHING.md)

## What was investigated?

1. Whether batching preserves logits when BatchNorm uses fixed running statistics.
2. How grouping a fixed workload changes CPU throughput.
3. Whether folding BatchNorm into adjacent linear layers preserves outputs and reduces execution time.

## Reproduce the procedure

1. Use a Python notebook environment with PyTorch and matplotlib. Dependency names are in requirements.txt; versions are intentionally unpinned because the original runtime versions were not recorded.
2. Supply the **same names.txt** used in the experiment, in the notebook working directory. It was not included in the upload, and its hash is unknown. A related learning dataset is available in the upstream makemore repository (https://github.com/karpathy/makemore), but equivalence to the original file has not been verified.
3. Open mlp_inference_study.ipynb in Jupyter or Colab. Use CPU tensors. Restart the kernel and run in order. Do not rerun only the split cell: it shuffles an already shuffled list.
4. Train the model, run diagnostic plots, switch to evaluation, check batching, construct folded layers, then run folding checks and benchmarks.
5. Preserve your new outputs as a separate run. Performance will vary with hardware and runtime conditions.

**Reproduction status:** code and saved outputs are available; this package was syntax-checked but not retrained or benchmarked during packaging. No trained checkpoint or dataset is bundled, so exact replay is not available from these files alone. The notebook contains the procedure needed to retrain when the data is supplied.

## Credits and ownership

The model follows the style of Andrej Karpathy's makemore educational work. Confirm the exact source/version and retain applicable upstream notices before publishing. The author ran the experiments and supplied their outputs. ChatGPT assisted with interpretation and documentation.
Note:This is a beginners study and the code is very out of place.The experiment is more of a personal study but the results are easy to understand and interpret so more beginners like me can learn from this small experiment.The model architecture is very basic for the same reason. Enjoy!! 
