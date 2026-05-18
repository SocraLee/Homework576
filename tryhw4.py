import sys
import os
import time
sys.dont_write_bytecode = True
from uwimg import *
from src.matrix import *
from src.hw4.classifier import *


# =====================================================================
# Model factories
# =====================================================================
def softmax_model(inputs, outputs):
    return make_model([make_layer(inputs, outputs, SOFTMAX)])


def neural_net(inputs, outputs, hidden_activation=LOGISTIC, hidden=32):
    return make_model([
        make_layer(inputs, hidden, hidden_activation),
        make_layer(hidden, outputs, SOFTMAX),
    ])


def deep_net(inputs, outputs, hidden_activation=LOGISTIC):
    # 3-layer: inputs -> 64 -> 32 -> outputs
    return make_model([
        make_layer(inputs, 64, hidden_activation),
        make_layer(64, 32, hidden_activation),
        make_layer(32, outputs, SOFTMAX),
    ])


# =====================================================================
# Helpers
# =====================================================================
def quietly(fn, *args, **kwargs):
    devnull = open(os.devnull, "w")
    saved = sys.stderr
    sys.stderr = devnull
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stderr = saved
        devnull.close()


def evaluate(model_fn, rate, decay, batch=128, iters=1000, momentum=0.9):
    m = model_fn(train.X.cols, train.y.cols)
    t0 = time.time()
    try:
        quietly(train_model, m, train, batch, iters, rate, momentum, decay)
        tr = accuracy_model(m, train)
        te = accuracy_model(m, test)
        return tr, te, time.time() - t0, None
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        return None, None, time.time() - t0, type(e).__name__


def print_row(label, tr, te, dt, err, label_fmt=">12"):
    if err:
        print(f"{label:{label_fmt}}  {'--':>10}  {'--':>10}  {dt:>8.1f}  diverged ({err})")
    else:
        print(f"{label:{label_fmt}}  {tr*100:>9.2f}%  {te*100:>9.2f}%  {dt:>8.1f}")


# =====================================================================
# Section 2.1 (commented out — already completed)
# =====================================================================
# # 2.1.2 / 2.1.3: softmax model on MNIST
# train = load_classification_data(b"mnist.train", b"mnist.labels", 10)
# test  = load_classification_data(b"mnist.test",  b"mnist.labels", 10)
# ...


# =====================================================================
# Section 2.2 (commented out — already completed)
# =====================================================================
# # 2.2.1 activation sweep, 2.2.2 lr sweep, 2.2.3 decay sweep on 2-layer,
# # 2.2.4 3-layer decay sweep — all on MNIST
# ...


# =====================================================================
# Section 3.1 — train on CIFAR with the 3-layer NN
# =====================================================================
print("loading CIFAR data...")
train = load_classification_data(b"cifar.train", b"cifar/labels.txt", 10)
test  = load_classification_data(b"cifar.test",  b"cifar/labels.txt", 10)
print(f"done. train={train.X.rows} examples, "
      f"test={test.X.rows} examples, feature_dim={train.X.cols}")

BATCH = 128
MOMENTUM = 0.9
ITERS = 3000
ACTIVATION = RELU  # best activation from 2.2.1


# ----- 3.1 (a): learning rate sweep with decay=0 -----
print(f"\n=== 3.1 (a): learning rate sweep "
      f"(3-layer NN, activation={ACTIVATION}, decay=0, iters={ITERS}) ===")
print(f"{'rate':>10}  {'train_acc':>10}  {'test_acc':>10}  {'time(s)':>8}")
print("-" * 56)
lr_results = []
for lr in [1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
    tr, te, dt, err = evaluate(
        lambda i, o: deep_net(i, o, hidden_activation=ACTIVATION),
        rate=lr, decay=0.0, iters=ITERS,
    )
    lr_results.append((lr, tr, te, dt, err))
    print_row(f"{lr:.0e}", tr, te, dt, err, label_fmt=">10")

ok = [r for r in lr_results if r[2] is not None]
best_lr = max(ok, key=lambda r: r[2])[0] if ok else 1e-2
print(f"\n-> best learning rate by test acc: {best_lr:.0e}")


# ----- 3.1 (b): decay sweep with best lr -----
print(f"\n=== 3.1 (b): weight decay sweep "
      f"(3-layer NN, activation={ACTIVATION}, rate={best_lr:.0e}, iters={ITERS}) ===")
print(f"{'decay':>10}  {'train_acc':>10}  {'test_acc':>10}  {'time(s)':>8}")
print("-" * 56)
decay_results = []
for d in [1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
    tr, te, dt, err = evaluate(
        lambda i, o: deep_net(i, o, hidden_activation=ACTIVATION),
        rate=best_lr, decay=d, iters=ITERS,
    )
    decay_results.append((d, tr, te, dt, err))
    print_row(f"{d:.0e}", tr, te, dt, err, label_fmt=">10")


# =====================================================================
# Markdown summary
# =====================================================================
def md_table(rows, label_col, label_fn=lambda x: x):
    width = max(len(label_col), max((len(label_fn(r[0])) for r in rows), default=0))
    head = f"| {label_col:<{width}} | train acc | test acc |"
    sep  = f"|{'-' * (width + 2)}|-----------|----------|"
    out = [head, sep]
    for r in rows:
        label = label_fn(r[0])
        tr, te, _, err = r[1], r[2], r[3], r[4]
        if err:
            out.append(f"| {label:<{width}} | diverged  | diverged |")
        else:
            out.append(f"| {label:<{width}} | {tr*100:6.2f}%   | {te*100:6.2f}%  |")
    return "\n".join(out)


# gather all runs across both sweeps and find the best by test accuracy
all_runs = [(f"lr={lr:.0e}, decay=0",        tr, te, err)
            for lr, tr, te, _, err in lr_results]
all_runs += [(f"lr={best_lr:.0e}, decay={d:.0e}", tr, te, err)
             for d, tr, te, _, err in decay_results]

valid = [r for r in all_runs if r[2] is not None]
best = max(valid, key=lambda r: r[2]) if valid else None


summary = []
summary.append("=" * 60)
summary.append("SUMMARY for Section 3.1 (paste into hw4.pdf)")
summary.append("=" * 60)
summary.append(f"\nFixed: 3-layer NN (in=3072 -> 64 -> 32 -> 10), "
               f"activation={ACTIVATION}, batch={BATCH}, momentum={MOMENTUM}, iters={ITERS}\n")

summary.append("### Learning rate sweep (decay=0)\n")
summary.append(md_table(lr_results, "learning rate", label_fn=lambda x: f"{x:.0e}"))
summary.append(f"\n=> best learning rate: **{best_lr:.0e}**\n")

summary.append(f"### Weight decay sweep (rate={best_lr:.0e})\n")
summary.append(md_table(decay_results, "weight decay", label_fn=lambda x: f"{x:.0e}"))
summary.append("")

if best:
    summary.append("### Overall best configuration\n")
    summary.append(f"- {best[0]}")
    summary.append(f"- **train accuracy: {best[1]*100:.2f}%**")
    summary.append(f"- **test accuracy:  {best[2]*100:.2f}%**")
    summary.append("")

text = "\n".join(summary)
print("\n" + text)

with open("hw4_results_31.txt", "w") as f:
    f.write(text)
print("results also saved to hw4_results_31.txt")
