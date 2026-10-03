# AI Engineering Foundations

Hands-on exercises for an AI engineering learning roadmap.

## Week 1

| Script | Topic |
|---|---|
| `week1/sentiment.py` | Concurrent LLM calls with `asyncio`, structured outputs, and pydantic validation |
| `week1/train.py` | A small PyTorch MLP trained on a synthetic two-moons dataset, with an annotated training loop |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then set OPENAI_API_KEY
```

## Run

```bash
uv run week1/train.py
uv run week1/sentiment.py
```

## Docs

Open `docs/flow-map.html` in a browser for an interactive diagram of both scripts: run order, the tool and technique behind each step, and a concurrency simulation.
