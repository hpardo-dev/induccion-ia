"""Entrena una red 784-16-16-10 (sigmoide + softmax) sobre MNIST con numpy.

Guarda instantáneas de los pesos durante el entrenamiento para que la
animación pueda mostrar la red "antes" y "después" de aprender.

Uso:  python3 training/train.py   (desde la raíz del repo)
Salida: web/model.js
"""
import gzip
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "web", "model.js")
rng = np.random.default_rng(7)


def load_images(name):
    with gzip.open(os.path.join(DATA, name)) as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255


def load_labels(name):
    with gzip.open(os.path.join(DATA, name)) as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


Xtr, ytr = load_images("train-images-idx3-ubyte.gz"), load_labels("train-labels-idx1-ubyte.gz")
Xte, yte = load_images("t10k-images-idx3-ubyte.gz"), load_labels("t10k-labels-idx1-ubyte.gz")


def shift_batch(xb):
    """Desplaza aleatoriamente cada imagen hasta ±1 px (la mitad de las veces) para que la red tolere
    dígitos dibujados a mano que no estén perfectamente centrados."""
    imgs = xb.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        dy, dx = rng.integers(-1, 2, size=2) if rng.random() < 0.5 else (0, 0)
        out[i] = np.roll(np.roll(im, dy, axis=0), dx, axis=1)
    return out.reshape(-1, 784)


sizes = [784, 16, 16, 10]
W = [rng.normal(0, 1 / np.sqrt(a), (b, a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
b = [np.zeros(n, np.float32) for n in sizes[1:]]

sig = lambda z: 1 / (1 + np.exp(-z))


def forward(x):
    a1 = sig(x @ W[0].T + b[0])
    a2 = sig(a1 @ W[1].T + b[1])
    z3 = a2 @ W[2].T + b[2]
    z3 -= z3.max(1, keepdims=True)
    p = np.exp(z3)
    p /= p.sum(1, keepdims=True)
    return a1, a2, p


def evaluate():
    _, _, p = forward(Xte)
    loss = float(-np.log(p[np.arange(len(yte)), yte] + 1e-9).mean())
    acc = float((p.argmax(1) == yte).mean())
    return loss, acc


# Adam
params = W + b
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
lr, beta1, beta2, eps = 3e-3, 0.9, 0.999, 1e-8
step = 0

snapshots = []
curve = []


def snapshot(label):
    loss, acc = evaluate()
    print(f"{label:>28}  loss={loss:.3f}  acc={acc:.3%}")
    snapshots.append({
        "label": label,
        "step": step,
        "loss": round(loss, 4),
        "acc": round(acc, 4),
        "W": [np.round(w, 3).tolist() for w in W],
        "b": [np.round(x, 3).tolist() for x in b],
    })


snapshot("Antes de entrenar")
snap_steps = {20: "Tras 20 lotes", 100: "Tras 100 lotes", 500: "Tras 500 lotes"}
EPOCHS, BS = 25, 64
n_batches = len(Xtr) // BS

for epoch in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for i in range(n_batches):
        idx = perm[i * BS:(i + 1) * BS]
        x, y = shift_batch(Xtr[idx]), ytr[idx]
        a1, a2, p = forward(x)
        d3 = p.copy()
        d3[np.arange(BS), y] -= 1
        d3 /= BS
        d2 = (d3 @ W[2]) * a2 * (1 - a2)
        d1 = (d2 @ W[1]) * a1 * (1 - a1)
        grads = [d1.T @ x, d2.T @ a1, d3.T @ a2, d1.sum(0), d2.sum(0), d3.sum(0)]
        step += 1
        for k, g in enumerate(grads):
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * g * g
            mh = m[k] / (1 - beta1 ** step)
            vh = v[k] / (1 - beta2 ** step)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)
        if step % 100 == 0 or step in (5, 10, 20, 50):
            curve.append([step, round(evaluate()[0], 4)])
        if step in snap_steps:
            snapshot(snap_steps[step])
    if epoch == 0:
        snapshot("Tras 1 época")
    if epoch == 4:
        snapshot("Tras 5 épocas")
    lr *= 0.9

snapshot(f"Final ({EPOCHS} épocas)")

# Ejemplos de test que la animación ofrece para probar: 3 por dígito,
# más algunos que la red final clasifica mal (útiles para hablar de errores).
_, _, p = forward(Xte)
pred = p.argmax(1)
examples = []
for d in range(10):
    for i in np.where((yte == d) & (pred == d))[0][:3]:
        examples.append({"label": int(d), "px": (Xte[i] * 255).astype(int).tolist(), "hard": False})
for i in np.where(pred != yte)[0][:6]:
    examples.append({"label": int(yte[i]), "px": (Xte[i] * 255).astype(int).tolist(), "hard": True})

model = {
    "sizes": sizes,
    "trainSize": len(Xtr),
    "testSize": len(Xte),
    "batchesPerEpoch": n_batches,
    "snapshots": snapshots,
    "curve": curve,
    "examples": examples,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("window.MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n")
print("guardado", OUT, os.path.getsize(OUT) // 1024, "KB")
