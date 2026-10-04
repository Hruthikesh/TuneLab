# TuneLab

When does fine-tuning actually beat RAG?

TuneLab is a Text-to-SQL experiment I built to compare how a small
language model performs under different ways of adapting it to a task.

The main comparison is:

- prompting
- few-shot prompting
- RAG
- LoRA
- QLoRA

The idea is not to build another chatbot. I wanted to measure what
actually changes when the training data, retrieval setup, and fine-tuning
strategy are changed.

## Why I built this

Fine-tuning and RAG are often discussed as alternatives, but the answer
depends on the task, amount of training data, retrieval quality, and
model.

Text-to-SQL gives me a controlled way to study this because the generated
SQL can be executed against the database and checked against the
reference query.

The main question I am investigating is:

> When does fine-tuning actually outperform RAG?

## Dataset

I use Spider 1.0.

- 7,000 training examples
- 1,034 development examples
- 166 databases
- multiple database domains

The training data is kept separate from the evaluation data throughout
the experiments.

Training subsets:

1%, 5%, 10%, 25%, 50%, 100%

## Models and methods

The experiments use:

Qwen2.5-Coder-1.5B-Instruct

I compare:

| Method | What changes |
|---|---|
| Zero-shot | Prompt only |
| Few-shot | Prompt + training examples |
| RAG | Retrieves relevant examples/schema information |
| LoRA | Parameter-efficient fine-tuning |
| QLoRA | 4-bit quantized fine-tuning + LoRA |

For LoRA I vary the rank.

For QLoRA I use NF4 quantization with double quantization.

## What I measure

The main metric is Execution Accuracy.

I also record:

- Exact Match
- SQL validity
- error categories
- inference latency
- training time
- GPU memory
- trainable parameters
- model/checkpoint size

For the fine-tuning experiments I also vary the amount of training data.

For RAG I vary the number of retrieved examples.

I also introduce controlled noise into the training data to see how
sensitive the different approaches are.

## Experiment design

The current experiment matrix contains 31 focused runs covering:

- method comparison
- training-data scaling
- RAG retrieval depth
- LoRA rank
- training-data noise

The exact configurations are stored in the experiment configuration
files rather than being hard-coded into the README.

## Evaluation

Every generated SQL query goes through the same evaluation pipeline.

For each example I check:

1. Is the SQL syntactically valid?
2. Can it be executed?
3. Does its result match the reference query?

I also classify failures into categories such as:

- wrong table
- wrong column
- wrong join
- incorrect filter
- aggregation
- GROUP BY
- ordering
- nested query
- syntax error
- schema misunderstanding

## Reproducibility

Each run records the configuration, random seed, model and dataset
information, hardware information, and experiment metadata.

Research results are generated from actual runs.

I don't include placeholder accuracy numbers in the repository.

## Current status

The complete pipeline and test suite are working.

Current local environment:

- Windows
- Python 3.11
- PyTorch with CUDA
- NVIDIA RTX 3050 Laptop GPU
- 6 GB VRAM

Spider has been downloaded and validated locally.

Current verification:

93 tests passed.

The full research experiment matrix has not been completed yet. Results
will be added after the GPU experiments are run.

## Project structure

```text
TuneLab/
├── configs/
├── data/
├── src/
│   └── tunelab/
│       ├── analysis/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── models/
│       ├── prompting/
│       ├── retrieval/
│       ├── training/
│       └── utils/
├── tests/
├── scripts/
├── experiments/
└── reports/