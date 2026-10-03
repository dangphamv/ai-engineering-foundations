import argparse
import math
from pathlib import Path

import torch
from torch import nn

from model import MoonsMLP

parser = argparse.ArgumentParser()
parser.add_argument("--hidden-size", type=int, default=32)
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "moons-mlp")
args = parser.parse_args()

torch.manual_seed(0)


def make_moons(n: int, noise: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    half = n // 2
    t = torch.rand(half) * math.pi
    upper = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    lower = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], dim=1)
    X = torch.cat([upper, lower]) + noise * torch.randn(2 * half, 2)
    y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
    return X, y


X, y = make_moons(2000, noise=0.25)
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
X_train, y_train = X[:1600], y[:1600]
X_val, y_val = X[1600:], y[1600:]

model = MoonsMLP(hidden_size=args.hidden_size, num_layers=args.num_layers)
num_params = sum(p.numel() for p in model.parameters())
print(f"MoonsMLP hidden_size={args.hidden_size} num_layers={args.num_layers} | {num_params:,} params")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

EPOCHS = args.epochs
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    # Bật chế độ training (quan trọng với Dropout/BatchNorm; ở đây không có nhưng nên tạo thói quen).
    model.train()
    # Xáo trộn thứ tự mẫu mỗi epoch để các batch khác nhau giữa các epoch -> gradient ít bị lệch.
    idx = torch.randperm(len(X_train))
    running_loss = 0.0

    for i in range(0, len(X_train), BATCH_SIZE):
        # Lấy ra một mini-batch theo chỉ số đã xáo trộn.
        batch = idx[i : i + BATCH_SIZE]
        xb, yb = X_train[batch], y_train[batch]

        # 1. Forward: đưa input qua mạng, nhận logits shape (batch, 2) — điểm thô chưa qua softmax.
        logits = model(xb)
        # 2. Tính loss: CrossEntropyLoss tự áp log-softmax rồi so với nhãn đúng yb.
        loss = loss_fn(logits, yb)

        # 3. Xoá gradient cũ: PyTorch CỘNG DỒN .grad qua các lần backward, nên phải reset mỗi bước.
        optimizer.zero_grad()
        # 4. Backward: autograd tính d(loss)/d(param) cho mọi tham số, lưu vào param.grad.
        loss.backward()
        # 5. Update: Adam dùng param.grad để cập nhật trọng số (param -= lr * bước_đã_điều_chỉnh).
        optimizer.step()

        # .item() lấy số Python ra khỏi tensor (tách khỏi graph); nhân batch size để tính trung bình đúng.
        running_loss += loss.item() * len(batch)

    # Chế độ đánh giá + tắt autograd: không cần gradient khi validate -> nhanh hơn, ít RAM hơn.
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = loss_fn(val_logits, y_val).item()
        # argmax trên chiều class -> nhãn dự đoán; so với nhãn thật để tính accuracy.
        val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        train_loss = running_loss / len(X_train)
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f} | val_acc {val_acc:.3f}"
        )

# Ghi config.json (kiến trúc) + model.safetensors (trọng số) — cùng format với model trên Hugging Face Hub.
model.save_pretrained(args.out)
print(f"saved to {args.out}")
