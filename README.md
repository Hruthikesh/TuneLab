# TuneLab

### When does fine-tuning actually beat RAG?

TuneLab is a project I built to understand the difference between RAG and
fine-tuning for Text-to-SQL.

I wanted to test this properly instead of just assuming that one approach
is better. So I am comparing zero-shot prompting, few-shot prompting, RAG,
LoRA and QLoRA using the Spider dataset.

## What I'm trying to find out

The main question is simple:

> When does fine-tuning actually become better than RAG?

The answer could change depending on how much training data is available,
how many examples RAG retrieves, the LoRA configuration and the amount of
noise in the data.

## Dataset

I'm using Spider 1.0 for the experiments.

- 7,000 training examples
- 1,034 development examples
- 166 databases

I created different training subsets from the training data:

- 1%
- 5%
- 10%
- 25%
- 50%
- 100%

The development set is kept separate from the training data.

## Methods

The project currently compares:

| Method | Setup |
|---|---|
| Zero-shot | Question + schema |
| Few-shot | Question + a few training examples |
| RAG | Retrieves relevant examples |
| LoRA | Fine-tunes the model with LoRA |
| QLoRA | 4-bit model + LoRA |

The base model I'm using is **Qwen2.5-Coder-1.5B-Instruct**.

For the experiments, I also change the RAG top-k value, LoRA rank,
training-data size and training-data noise.

## Evaluation

I don't want to judge the models only by whether their SQL looks similar
to the reference query.

The generated SQL is executed against the database and the result is
compared with the reference result.

I record:

- Execution Accuracy
- Exact Match
- SQL validity
- inference time
- training time
- GPU memory
- trainable parameters
- common SQL errors

Some of the error categories include wrong tables, wrong columns, joins,
filters, aggregation, ordering and syntax errors.

## Current status

The main pipeline is implemented and the test suite is passing.

The Spider dataset has been downloaded and checked locally, and the project
is set up to run the experiments on my RTX 3050 Laptop GPU.

The actual research experiments are still being run, so there are no
made-up results in this repository.

## Project structure

```text
TuneLab/
├── configs/
├── data/
├── docs/
├── experiments/
├── reports/
├── scripts/
├── src/
│   └── tunelab/
└── tests/
