import asyncio
import os
import time
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CONCURRENCY = 5

REVIEWS = [
    "Absolutely love this keyboard, typing feels amazing.",
    "Broke after two days. Total waste of money.",
    "It's fine. Does what it says, nothing special.",
    "Shipping was slow but the product quality is excellent.",
    "Worst customer service I've ever dealt with.",
    "Đồ dùng ổn, giá hơi cao nhưng chấp nhận được.",
]


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)


# Strict structured outputs force JSON matching this schema;
# pydantic still validates on our side (e.g. the 0..1 range isn't enforced by the API).
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}

client = AsyncOpenAI()
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def classify(text: str) -> Sentiment:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=MODEL,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "sentiment", "strict": True, "schema": SENTIMENT_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Classify the sentiment of this product review. "
                    f"confidence is a number between 0 and 1.\n\n{text}",
                }
            ],
        )
    choice = resp.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"refused: {choice.message.refusal}")
    return Sentiment.model_validate_json(choice.message.content)


async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        *(classify(r) for r in REVIEWS), return_exceptions=True
    )
    elapsed = time.perf_counter() - start

    for review, result in zip(REVIEWS, results):
        if isinstance(result, ValidationError):
            print(f"[INVALID] {review}\n  {result.errors()}")
        elif isinstance(result, Exception):
            print(f"[ERROR]   {review}\n  {type(result).__name__}: {result}")
        else:
            print(f"[{result.label:>8} {result.confidence:.2f}] {review}")
    print(f"\n{len(REVIEWS)} reviews in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
