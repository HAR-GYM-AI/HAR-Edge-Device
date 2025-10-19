"""Visualize live session predictions and sensor windows.

Usage:
    python scripts/visualize_live_session.py \
        --pred-file ./live_collection_data/live_predictions_20251019_115014.json \
        --sensor-file ./live_collection_data/live_sensor_data_20251019_115014.json \
        --out-dir visualizations

This script will produce:
- A PNG time-series plot with prediction confidences and selected sensor stats
- A JSON summary with basic statistics about predictions and latencies

Requires matplotlib, pandas, numpy
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_predictions_df(pred_json):
    rows = []
    for p in pred_json.get('predictions', []):
        row = {
            'timestamp_ms': p.get('timestamp_ms'),
            'exercise': p.get('exercise'),
            'confidence': p.get('confidence'),
        }
        # Add individual class scores if present
        scores = p.get('scores', {})
        for k, v in scores.items():
            row[f'score_{k}'] = v
        rows.append(row)
    return pd.DataFrame(rows)


def build_sensor_df(sensor_json):
    # We will extract qw_mean_* for each node and align by window_start_ms
    rows = []
    for win in sensor_json.get('sensor_windows', []):
        start = win.get('window_start_ms')
        entry = {'window_start_ms': start}
        sw = win.get('sensor_windows', {})
        for node_name, node_data in sw.items():
            feats = node_data.get('features', {})
            # choose means
            for comp in ('qw_mean', 'qx_mean', 'qy_mean', 'qz_mean', 'sma'):
                key = f'{comp}_{node_name}'
                entry[key] = feats.get(f'{comp}_{node_name}')
        rows.append(entry)
    return pd.DataFrame(rows)


def summarize_predictions(df):
    summary = {}
    if df.empty:
        return summary
    summary['total_predictions'] = len(df)
    summary['most_common'] = df['exercise'].mode().tolist() if 'exercise' in df else []
    summary['confidence_mean'] = float(df['confidence'].mean())
    summary['confidence_std'] = float(df['confidence'].std())
    # class counts
    summary['class_counts'] = df['exercise'].value_counts().to_dict() if 'exercise' in df else {}
    return summary


def plot_session(pred_df, sensor_df, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: class scores / confidence
    if not pred_df.empty:
        t = (pred_df['timestamp_ms'] - pred_df['timestamp_ms'].iloc[0]) / 1000.0
        ax0 = ax[0]
        # Plot overall confidence
        ax0.plot(t, pred_df['confidence'], label='pred_confidence', marker='o')
        # Plot any class scores columns
        score_cols = [c for c in pred_df.columns if c.startswith('score_')]
        for c in score_cols:
            ax0.plot(t, pred_df[c], label=c)
        ax0.set_ylabel('Confidence / Score')
        ax0.legend(loc='upper left')
        ax0.set_title('Predictions')

    # Bottom: sensor quaternion means for WRIST (if present)
    if not sensor_df.empty:
        ax1 = ax[1]
        # align time base
        t2 = (sensor_df['window_start_ms'] - sensor_df['window_start_ms'].iloc[0]) / 1000.0
        # try WRIST first, else first available node column
        wrist_cols = [c for c in sensor_df.columns if c.endswith('_WRIST') and c.startswith('qw_mean')]
        if wrist_cols:
            ax1.plot(t2, sensor_df[wrist_cols[0]], label=wrist_cols[0])
        # plot SMA for any node
        sma_cols = [c for c in sensor_df.columns if c.startswith('sma_')]
        for c in sma_cols:
            ax1.plot(t2, sensor_df[c], label=c)
        ax1.set_ylabel('Sensor mean / SMA')
        ax1.set_xlabel('Seconds (relative)')
        ax1.legend(loc='upper left')
        ax1.set_title('Sensor Window Statistics')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_latency_stats(pred_json):
    meta = pred_json.get('session_metadata', {})
    return meta.get('latency_stats', {})


def detailed_stats(pred_json, pred_df):
    stats = {}
    # total reps from metadata if present
    stats['total_reps'] = pred_json.get('session_metadata', {}).get('total_reps', None)

    # per-prediction latencies: collect keys
    lat_keys = set()
    lat_values = {}
    for p in pred_json.get('predictions', []):
        l = p.get('latency_ms', {})
        for k, v in l.items():
            lat_keys.add(k)
            lat_values.setdefault(k, []).append(v)

    # summarize latencies
    stats['latency_stage_stats'] = {}
    import math
    for k in sorted(lat_keys):
        arr = lat_values.get(k, [])
        if not arr:
            continue
        stats['latency_stage_stats'][k] = {
            'mean': float(np.mean(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'std': float(np.std(arr, ddof=0))
        }

    # per-class mean/std confidences
    classes = pred_df['exercise'].unique().tolist() if not pred_df.empty else []
    stats['per_class_confidence'] = {}
    for c in classes:
        vals = pred_df[pred_df['exercise'] == c]['confidence']
        stats['per_class_confidence'][c] = {
            'mean': float(vals.mean()),
            'std': float(vals.std())
        }

    # total detected reps across predictions
    stats['reps_detected_total'] = int(sum(p.get('reps_detected', 0) for p in pred_json.get('predictions', [])))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-file', required=True)
    parser.add_argument('--sensor-file', required=True)
    parser.add_argument('--out-dir', default='visualizations')
    args = parser.parse_args()

    pred = load_json(args.pred_file)
    sensor = load_json(args.sensor_file)

    pred_df = build_predictions_df(pred)
    sensor_df = build_sensor_df(sensor)

    summary = summarize_predictions(pred_df)
    summary['latency_stats'] = compute_latency_stats(pred)

    os.makedirs(args.out_dir, exist_ok=True)
    # create filenames based on session start
    start_iso = pred.get('session_metadata', {}).get('start_time', datetime.utcnow().isoformat())
    start_tag = start_iso.replace(':', '').replace('-', '').replace('.', '').replace('T', '_')
    out_png = os.path.join(args.out_dir, f'live_session_{start_tag}.png')
    out_json = os.path.join(args.out_dir, f'live_session_summary_{start_tag}.json')
    out_csv = os.path.join(args.out_dir, f'live_session_stats_{start_tag}.csv')

    plot_session(pred_df, sensor_df, out_png)

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # Build a small statistics table (CSV)
    rows = []
    rows.append({'stat': 'total_predictions', 'value': summary.get('total_predictions', 0)})
    rows.append({'stat': 'confidence_mean', 'value': summary.get('confidence_mean')})
    rows.append({'stat': 'confidence_std', 'value': summary.get('confidence_std')})
    # per-class counts and mean confidences
    class_counts = summary.get('class_counts', {})
    for cls, cnt in class_counts.items():
        rows.append({'stat': f'class_count_{cls}', 'value': cnt})
        # mean class confidence from pred_df if available
        if f'score_{cls}' in pred_df.columns:
            rows.append({'stat': f'class_mean_score_{cls}', 'value': float(pred_df[f'score_{cls}'].mean())})

    import csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stat', 'value'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Saved plot to', out_png)
    print('Saved summary to', out_json)

    # Detailed stats and table
    det = detailed_stats(pred, pred_df)
    out_json_detailed = os.path.join(args.out_dir, f'live_session_summary_detailed_{start_tag}.json')
    with open(out_json_detailed, 'w') as f:
        json.dump({'summary': summary, 'detailed': det}, f, indent=2)

    # Also write a table CSV of interesting fields
    out_table_csv = os.path.join(args.out_dir, f'live_session_table_{start_tag}.csv')
    table_rows = []
    # session-level fields
    table_rows.append({'field': 'total_predictions', 'value': summary.get('total_predictions')})
    table_rows.append({'field': 'total_sensor_windows', 'value': pred.get('session_metadata', {}).get('total_sensor_windows')})
    table_rows.append({'field': 'total_reps_meta', 'value': pred.get('session_metadata', {}).get('total_reps')})
    table_rows.append({'field': 'reps_detected_total', 'value': det.get('reps_detected_total')})

    # latency stages
    for k, v in det.get('latency_stage_stats', {}).items():
        table_rows.append({'field': f'latency_{k}_mean', 'value': v['mean']})
        table_rows.append({'field': f'latency_{k}_min', 'value': v['min']})
        table_rows.append({'field': f'latency_{k}_max', 'value': v['max']})
        table_rows.append({'field': f'latency_{k}_std', 'value': v['std']})

    # per-class confidences
    for c, v in det.get('per_class_confidence', {}).items():
        table_rows.append({'field': f'class_{c}_mean_conf', 'value': v['mean']})
        table_rows.append({'field': f'class_{c}_std_conf', 'value': v['std']})

    with open(out_table_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['field', 'value'])
        writer.writeheader()
        for r in table_rows:
            writer.writerow(r)

    print('Saved detailed summary to', out_json_detailed)
    print('Saved stats CSV to', out_csv)
    print('Saved table CSV to', out_table_csv)


if __name__ == '__main__':
    main()
