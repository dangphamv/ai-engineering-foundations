import argparse
from pathlib import Path

import torch

from model import MoonsMLP

parser = argparse.ArgumentParser()
parser.add_argument("x", type=float)
parser.add_argument("y", type=float)
parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "moons-mlp")
args = parser.parse_args()

model = MoonsMLP.from_pretrained(args.model)
model.eval()

with torch.no_grad():
    probs = model(torch.tensor([[args.x, args.y]])).softmax(dim=1)[0]

label = int(probs.argmax())
print(f"({args.x}, {args.y}) -> class {label} ({'upper' if label == 0 else 'lower'} moon) | p={probs[label]:.3f}")
