"""Demo edukasi: backpropagation & gradient descent TANPA PyTorch autograd.

Model terkecil: y_hat = w*x + b, loss MSE. Kita hitung gradien dengan tangan
(aturan rantai), lalu bandingkan dengan autograd PyTorch.
Jalankan: python -m ml.scripts.gradient_demo
"""
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([3.0, 5.0, 7.0, 9.0])     # target: y = 2x + 1

# --- Versi manual ---
w, b, lr = 0.0, 0.0, 0.05
for step in range(200):
    y_hat = w * x + b                        # FORWARD
    loss = ((y_hat - y) ** 2).mean()         # LOSS (MSE)
    # BACKWARD (aturan rantai):
    #   dL/dy_hat = 2*(y_hat-y)/N ; dy_hat/dw = x ; dy_hat/db = 1
    dw = (2 * (y_hat - y) * x).mean()
    db = (2 * (y_hat - y)).mean()
    w, b = w - lr * dw.item(), b - lr * db.item()   # GRADIENT DESCENT
    if step % 40 == 0:
        print(f"step {step:3d} | loss {loss.item():.5f} | w {w:.4f} b {b:.4f}")
print(f"Hasil manual : w={w:.4f}, b={b:.4f} (target 2, 1)")

# --- Verifikasi dengan autograd ---
wt = torch.tensor(0.0, requires_grad=True)
bt = torch.tensor(0.0, requires_grad=True)
loss = ((wt * x + bt - y) ** 2).mean()
loss.backward()
manual_dw = (2 * (0.0 * x + 0.0 - y) * x).mean()
print(f"Cek gradien awal: autograd dw={wt.grad.item():.4f} vs manual dw={manual_dw.item():.4f}")
