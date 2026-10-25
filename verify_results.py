"""Verify the included data, full histories and inference checkpoints on CPU."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    report = []
    for task, filename in [('addition', 'mod97.npz'), ('multiplication', 'mul97.npz')]:
        folder = ROOT / task
        sys.path.insert(0, str(folder))
        sys.modules.pop('data', None)
        with np.load(folder / filename, allow_pickle=False) as dataset:
            p = int(dataset['p'])
            assert p == 97
            pairs = np.concatenate([dataset['x_train'], dataset['x_test']])
            labels = np.concatenate([dataset['y_train'], dataset['y_test']])
            expected = (pairs[:, 0] + pairs[:, 1]) % p if task == 'addition' else (pairs[:, 0] * pairs[:, 1]) % p
            assert np.array_equal(labels, expected)
            assert len(pairs) == p * p == len(np.unique(pairs, axis=0))
            assert np.all((pairs >= 0) & (pairs < p))
            assert len(dataset['x_train']) == 2822 and len(dataset['x_test']) == 6587
            order = np.random.default_rng(2026).permutation(p * p)
            all_pairs = np.array([(a, b) for a in range(p) for b in range(p)])
            assert np.array_equal(pairs, all_pairs[order])
            for model_name, class_name in [('mlp', 'TinyMLP'), ('transformer', 'TinyTransformer')]:
                spec = importlib.util.spec_from_file_location(f'{task}_{model_name}', folder / f'{model_name}.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                model = getattr(module, class_name)(p).eval()
                run = folder / 'results' / model_name / 'seed_42'
                checkpoint = torch.load(run / 'epoch_30000.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model'])
                with (run / 'history.csv').open(newline='') as f:
                    rows = list(csv.DictReader(f))
                epochs = [int(r['epoch']) for r in rows]
                expected_epochs = [1] + list(range(1000, 30001, 1000)) if task == 'addition' and model_name == 'transformer' else list(range(1, 30001))
                assert epochs == expected_epochs
                assert all(np.isfinite(float(r[k])) for r in rows for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc'])
                with (run.parent / 'summary.csv').open(newline='') as f:
                    summary = next(csv.DictReader(f))
                first95 = next(int(r['epoch']) for r in rows if float(r['test_acc']) >= 95)
                assert first95 == int(summary['first_observed_95pct_epoch'])
                result = {'task': task, 'model': model_name, 'seed': 42, 'final_epoch': epochs[-1], 'history_rows': len(rows), 'first_observed_test_95_epoch': first95}
                assert checkpoint['epoch'] == 30000 and checkpoint['seed'] == 42
                for split in ['train', 'test']:
                    x = torch.from_numpy(dataset[f'x_{split}']).long()
                    y = torch.from_numpy(dataset[f'y_{split}']).long()
                    with torch.inference_mode():
                        logits = torch.cat([model(batch) for batch in x.split(1024)])
                        correct = int((logits.argmax(1) == y).sum())
                        loss = float(torch.nn.functional.cross_entropy(logits, y))
                    accuracy = 100 * correct / len(y)
                    assert abs(accuracy - float(rows[-1][f'{split}_acc'])) < 1e-4
                    assert abs(accuracy - float(summary[f'final_{split}_acc'])) < 1e-4
                    assert abs(loss - float(rows[-1][f'{split}_loss'])) < 1e-4
                    result[f'{split}_correct'] = correct
                    result[f'{split}_count'] = len(y)
                    result[f'{split}_accuracy_pct'] = accuracy
                    result[f'{split}_loss'] = loss
                report.append(result)
                print(json.dumps(result), flush=True)
        sys.path.pop(0)
    (ROOT / 'verification.json').write_text(json.dumps({'device': 'cpu', 'torch_version': torch.__version__, 'results': report}, indent=2) + '\n', encoding='utf-8')
    print('All checks passed. Saved verification.json')


if __name__ == '__main__':
    main()
