# AI Engineering Foundations

Hands-on exercises for an AI engineering learning roadmap.

## Week 1

| Script | Topic |
|---|---|
| `week1/sentiment.py` | Concurrent LLM calls with `asyncio`, structured outputs, and pydantic validation |
| `week1/train.py` | Trains `MoonsMLP` on a synthetic two-moons dataset (annotated training loop) and saves it in Hugging Face format |
| `week1/model.py` | `MoonsMLP`: configurable MLP with `save_pretrained` / `from_pretrained` |
| `week1/predict.py` | Loads a saved model and classifies a point |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then set OPENAI_API_KEY
```

## Run

```bash
uv run week1/train.py
uv run week1/predict.py 0 1
uv run week1/sentiment.py
```

## Models

Saved under `models/` as `config.json` + `model.safetensors`, the layout used by models on the Hugging Face Hub.

| Model | Config | Params | Val acc |
|---|---|---|---|
| `models/moons-mlp` | hidden 32, 2 layers | 1,218 | 95.5% |
| `models/moons-mlp-1m` | hidden 512, 6 layers | 1,315,842 | 95.2% |

Train a different size:

```bash
uv run week1/train.py --hidden-size 512 --num-layers 6 --lr 1e-3 --out models/moons-mlp-1m
uv run week1/predict.py 0 1 --model models/moons-mlp-1m
```

## Docs

Open `docs/flow-map.html` in a browser for an interactive guide: run order, PyTorch and pydantic basics, the tool and technique behind each step, a step-by-step training walkthrough with real numbers, and the trained model running in the browser.
