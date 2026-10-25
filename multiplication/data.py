"""Shared data, experiment runner and plotting for modular-multiplication Grokking."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

P = 97
SEEDS = [42, 123, 2026, 3407, 7777]


def generate(path='mul97.npz', p=P, train_ratio=0.3, split_seed=2026):
    rng = np.random.default_rng(split_seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] * pairs[:, 1]) % p
    order = rng.permutation(len(pairs))
    n = int(len(pairs) * train_ratio)
    np.savez(path, x_train=pairs[order[:n]], y_train=labels[order[:n]],
             x_test=pairs[order[n:]], y_test=labels[order[n:]], p=p,
             train_ratio=train_ratio, split_seed=split_seed)
    print(f'Saved {path}: train={n}, test={len(pairs)-n}, p={p}, split_seed={split_seed}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mul97.npz')
    parser.add_argument('--epochs', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results')
    return parser.parse_args()


def pick_device(pref='auto'):
    """'auto' prefers CUDA, falls back to CPU."""
    if pref == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(pref)


def save_checkpoint(path, model, opt, epoch, seed, metrics):
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
             'epoch': epoch, 'seed': seed, 'metrics': metrics,
             'torch_rng': torch.get_rng_state(),
             'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate()}
    if torch.cuda.is_available():
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def draw_curves(rows, out):
    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax, label in [('acc', axes[0], 'Accuracy (%)'), ('loss', axes[1], 'Cross entropy')]:
        ax.plot(epochs, [r['train_' + metric] for r in rows], label='Train')
        ax.plot(epochs, [r['test_' + metric] for r in rows], label='Test')
        ax.set(xlabel='Epoch', ylabel=label)
        ax.legend()
        ax.grid(alpha=.2)
    axes[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run_experiment(model_name, make_model, args):
    if not Path(args.data).exists():
        raise FileNotFoundError(f'{args.data} does not exist; first run: python data.py')
    data = np.load(args.data)
    p = int(data['p'])
    dev = pick_device(args.device)
    gpu_name = f' ({torch.cuda.get_device_name(dev)})' if dev.type == 'cuda' else ''
    print(f'Training device: {dev}{gpu_name}', flush=True)
    x_train = torch.tensor(data['x_train'], dtype=torch.long, device=dev)
    y_train = torch.tensor(data['y_train'], dtype=torch.long, device=dev)
    x_test = torch.tensor(data['x_test'], dtype=torch.long, device=dev)
    y_test = torch.tensor(data['y_test'], dtype=torch.long, device=dev)
    root = Path(args.out) / model_name
    root.mkdir(parents=True, exist_ok=True)
    summary = []

    for seed in args.seeds:
        seed_everything(seed)
        model = make_model(p).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        run_dir = root / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        hit95 = None
        start = time.perf_counter()
        print(f'[{model_name}] seed={seed} device={dev} parameters={sum(t.numel() for t in model.parameters()):,}', flush=True)

        # Full-batch training; metrics are recorded EVERY epoch so the curves are dense,
        # but they are only printed every --log-every epochs.
        for epoch in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train), y_train)
            loss.backward()
            opt.step()
            model.eval()
            with torch.inference_mode():
                tr_logits = model(x_train)
                te_logits = model(x_test)
                metrics = {
                    'epoch': epoch,
                    'train_loss': float(F.cross_entropy(tr_logits, y_train).item()),
                    'test_loss': float(F.cross_entropy(te_logits, y_test).item()),
                    'train_acc': float((tr_logits.argmax(1) == y_train).float().mean().item() * 100),
                    'test_acc': float((te_logits.argmax(1) == y_test).float().mean().item() * 100),
                    'weight_norm': float(torch.sqrt(sum((z.detach() ** 2).sum() for z in model.parameters())).item()),
                    'elapsed_sec': round(time.perf_counter() - start, 2),
                }
            rows.append(metrics)
            if hit95 is None and metrics['test_acc'] >= 95:
                hit95 = epoch  # first epoch whose test accuracy reaches 95%
            should_log = epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs
            if should_log:
                print(f"  epoch={epoch:6d} train_loss={metrics['train_loss']:.4f} test_loss={metrics['test_loss']:.4f} "
                      f"train={metrics['train_acc']:6.2f}% test={metrics['test_acc']:6.2f}% "
                      f"elapsed={metrics['elapsed_sec']:.1f}s", flush=True)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(run_dir / f'epoch_{epoch}.pt', model, opt, epoch, seed, metrics)

        total_sec = time.perf_counter() - start
        print(f'[{model_name}] seed={seed} finished {args.epochs} epochs in {total_sec:.1f}s', flush=True)

        with (run_dir / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        draw_curves(rows, run_dir / 'curves.png')
        item = {'model': model_name, 'seed': seed, 'parameters': sum(t.numel() for t in model.parameters()),
                'first_observed_95pct_epoch': hit95, 'final_test_acc': rows[-1]['test_acc'],
                'final_train_acc': rows[-1]['train_acc'], 'elapsed_sec': rows[-1]['elapsed_sec']}
        summary.append(item)

    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f'Completed. Results saved in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create deterministic modular-multiplication dataset')
    parser.add_argument('--p', type=int, default=97)
    parser.add_argument('--train-ratio', type=float, default=0.3)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--out', default='mul97.npz')
    a = parser.parse_args()
    if a.p < 2 or not 0 < a.train_ratio < 1:
        parser.error('Require --p >= 2 and 0 < --train-ratio < 1')
    generate(a.out, a.p, a.train_ratio, a.split_seed)
