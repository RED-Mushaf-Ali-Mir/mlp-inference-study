# Publishing and application guide

## Recommended order

1. Publish the study on GitHub as the canonical code-and-evidence page.
2. Link it from your portfolio's projects section.
3. Share a short LinkedIn post linking to the repository.
4. Link the repository beside this project in your CV. Use it as evidence of relevant preparation in a motivation statement, not as a claimed publication.

A journal submission or arXiv preprint is not recommended for this small educational study alone. It implements a known optimization and does not establish a new algorithm or broad empirical claim.

## GitHub setup

Suggested repository: mlp-inference-study
Description: CPU inference experiments on a character-level MLP: batching, BatchNorm folding, correctness checks, and measurement variability.
Topics: pytorch, machine-learning-systems, inference, benchmarking, batch-normalization

Create a public repository through GitHub's New repository page. Upload the extracted contents of this package so README.md is at the repository root; do not upload just the ZIP. Preserve the results directory. Pin the repository on your profile. If you already maintain the character-language-modeling repository, an equally good option is to place this study under experiments/mlp-inference and link it prominently from the main README; maintain one canonical copy.

Before public upload, confirm the exact learning source and applicable license, read the notebook outputs for anything you would not want public, and ensure the README's missing-data/checkpoint/version limitations remain visible. Public posting was not performed by the assistant.

## LinkedIn draft

I finished a small CPU inference study using my character-level MLP.

I compared batching and then folded six Linear–BatchNorm pairs into linear layers for inference. At batch size 32, the fixed 128-context workload ran about 1.87x faster in both measurement rounds. Across 22,767 validation contexts, the numerical checks passed and no highest-scoring predictions changed.

The measurements also taught me to be careful with speedup claims: some configurations had substantial timing variability, and the benefit at batch size 128 was inconclusive.

This is a learning study of a known optimization, building on Karpathy's makemore, with ChatGPT-assisted experiment development. I have included the code, results and limitations in the repository.

Repository: [insert the published GitHub URL]

## CV wording

Investigated CPU inference in a character-level MLP using batching and BatchNorm folding; observed a 1.87x speedup at batch size 32 in two rounds and validated numerical agreement on 22,767 contexts, reporting measurement variability and reproducibility limits.

Place this under Projects or Independent Studies, not Publications. Retain learning-source attribution in the linked repository. Be ready to explain the folding algebra, column-wise scaling, evaluation-mode requirement, measured workload, and why the noisy largest speedup was not selected as the headline.

## ETH presentation

Connect the project to an interest in efficient neural-network execution and careful systems measurement. Do not imply that posting on LinkedIn or obtaining GitHub stars is a selection criterion. ETH asks for relevant background, documents, and a motivation statement; a clear artifact provides supporting evidence, not a guarantee of admission.

Official references checked while packaging:
- https://inf.ethz.ch/studies/summer-research-fellowship/how-to-apply.html
- https://docs.github.com/en/repositories/creating-and-managing-repositories
- https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile
