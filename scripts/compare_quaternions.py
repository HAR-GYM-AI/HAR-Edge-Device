"""Compare quaternion means between two sessions.

Usage:
    python3 scripts/compare_quaternions.py \
        --session_a_preds live_collection_data/live_predictions_20251019_115014.json \
        --session_a_sensors live_collection_data/live_sensor_data_20251019_115014.json \
        --session_b_preds live_collection_data/live_predictions_20251019_135650.json \
        --session_b_sensors live_collection_data/live_sensor_data_135650.json \
        --out visualizations/compare_pullup_bicepcurl.png

The script will:
- select first N windows predicted as PULL_UP from session A
- select first N windows predicted as BICEP_CURL from session B
- plot per-node quaternion mean components (qw,qx,qy,qz) side-by-side
"""
import argparse
import json
import os
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def map_predictions_to_windows(pred_json, sensor_json):
    # Build list of window_start_ms
    windows = sensor_json.get('sensor_windows', [])
    win_starts = [w.get('window_start_ms') for w in windows]

    mapping = []  # list of (prediction, window_index)
    for p in pred_json.get('predictions', []):
        ts = p.get('timestamp_ms')
        # find nearest window index
        if not win_starts:
            mapping.append((p, None))
            continue
        diffs = [abs((ws - ts)) for ws in win_starts]
        idx = int(np.argmin(diffs))
        mapping.append((p, idx))
    return mapping


def extract_quat_means_for_indices(sensor_json, indices, nodes=('WRIST','BICEP','CHEST','THIGH')):
    # returns dict[node] -> list of (qw,qx,qy,qz,sma)
    out = {n: [] for n in nodes}
    windows = sensor_json.get('sensor_windows', [])
    for i in indices:
        if i is None or i < 0 or i >= len(windows):
            for n in nodes:
                out[n].append((np.nan,)*5)
            continue
        sw = windows[i].get('sensor_windows', {})
        for n in nodes:
            feats = sw.get(n, {}).get('features', {})
            qw = feats.get(f'qw_mean_{n}')
            qx = feats.get(f'qx_mean_{n}')
            qy = feats.get(f'qy_mean_{n}')
            qz = feats.get(f'qz_mean_{n}')
            sma = feats.get(f'sma_{n}')
            out[n].append((qw, qx, qy, qz, sma))
    return out


def plot_comparison(a_quats, b_quats, nodes, out_path, n_windows):
    comps = ['qw','qx','qy','qz']
    fig, axes = plt.subplots(len(nodes), len(comps), figsize=(14, 3*len(nodes)), sharex=True)
    if len(nodes) == 1:
        axes = [axes]
    for r, node in enumerate(nodes):
        a_arr = np.array(a_quats[node], dtype=float)
        b_arr = np.array(b_quats[node], dtype=float)
        for c, comp in enumerate(comps):
            ax = axes[r][c] if len(nodes) > 1 else axes[0][c]
            # component index
            ax.plot(range(n_windows), a_arr[:, c], label='session_A_pullup', marker='o')
            ax.plot(range(n_windows), b_arr[:, c], label='session_B_bicep_as_pullup', marker='x')
            ax.set_title(f'{node} {comp}')
            ax.grid(True)
            if r == 0 and c == 0:
                ax.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_a_preds', required=True)
    parser.add_argument('--session_a_sensors', required=True)
    parser.add_argument('--session_b_preds', required=True)
    parser.add_argument('--session_b_sensors', required=True)
    parser.add_argument('--n', type=int, default=5, help='number of windows to compare')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    a_pred = load_json(args.session_a_preds)
    a_sens = load_json(args.session_a_sensors)
    b_pred = load_json(args.session_b_preds)
    b_sens = load_json(args.session_b_sensors)

    a_map = map_predictions_to_windows(a_pred, a_sens)
    b_map = map_predictions_to_windows(b_pred, b_sens)

    # select first N pull_up windows from A
    a_indices = [idx for (p, idx) in a_map if p.get('exercise') == 'PULL_UP'][:args.n]
    # select first N bicep_curl windows from B
    b_indices = [idx for (p, idx) in b_map if p.get('exercise') == 'BICEP_CURL'][:args.n]

    # if not enough windows, pad or truncate to same length
    n_windows = max(len(a_indices), len(b_indices))
    if len(a_indices) < n_windows:
        a_indices += [None] * (n_windows - len(a_indices))
    if len(b_indices) < n_windows:
        b_indices += [None] * (n_windows - len(b_indices))

    nodes = ['WRIST','BICEP','CHEST','THIGH']
    a_quats = extract_quat_means_for_indices(a_sens, a_indices, nodes=nodes)
    b_quats = extract_quat_means_for_indices(b_sens, b_indices, nodes=nodes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_comparison(a_quats, b_quats, nodes, args.out, n_windows)

    print('Saved comparison plot to', args.out)


if __name__ == '__main__':
    main()
