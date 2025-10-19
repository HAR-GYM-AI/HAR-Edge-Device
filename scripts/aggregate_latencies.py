"""Aggregate latency stats across all live prediction sessions.

Produces JSON and CSV outputs in `visualizations/` and prints a summary.
"""
import glob
import json
import os
from statistics import mean


def collect_files(pattern="live_collection_data/live_predictions_*.json"):
    return sorted(glob.glob(pattern))


def safe_get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def aggregate(files):
    types = ['processing_latency_ms', 'data_collection_latency_ms', 'total_latency_ms']
    # For each type, collect lists of session-level {mean,min,max}
    data = {t: {'means': [], 'mins': [], 'maxs': []} for t in types}

    sessions = []
    for f in files:
        try:
            j = json.load(open(f, 'r'))
        except Exception:
            continue
        meta = j.get('session_metadata', {})
        lat = meta.get('latency_stats', {})
        sessions.append(os.path.basename(f))
        for t in types:
            v = lat.get(t)
            if isinstance(v, dict):
                data[t]['means'].append(v.get('mean'))
                data[t]['mins'].append(v.get('min'))
                data[t]['maxs'].append(v.get('max'))

    # Compute overall aggregates
    out = {}
    for t in types:
        arr_means = [x for x in data[t]['means'] if x is not None]
        arr_mins = [x for x in data[t]['mins'] if x is not None]
        arr_maxs = [x for x in data[t]['maxs'] if x is not None]
        if not arr_means:
            out[t] = None
            continue
        out[t] = {
            'overall_min_of_mins': min(arr_mins) if arr_mins else None,
            'overall_mean_of_means': mean(arr_means) if arr_means else None,
            'overall_max_of_maxs': max(arr_maxs) if arr_maxs else None,
            'avg_session_min': mean(arr_mins) if arr_mins else None,
            'avg_session_mean': mean(arr_means) if arr_means else None,
            'avg_session_max': mean(arr_maxs) if arr_maxs else None,
            'num_sessions': len(arr_means)
        }

    return {'files': sessions, 'aggregates': out}


def main():
    files = collect_files()
    if not files:
        print('No files found')
        return
    res = aggregate(files)
    os.makedirs('visualizations', exist_ok=True)
    out_json = 'visualizations/all_sessions_latency_aggregate.json'
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    # also write CSV
    import csv
    out_csv = 'visualizations/all_sessions_latency_aggregate.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['latency_type', 'overall_min_of_mins', 'overall_mean_of_means', 'overall_max_of_maxs', 'avg_session_min', 'avg_session_mean', 'avg_session_max', 'num_sessions'])
        for k, v in res['aggregates'].items():
            if v is None:
                writer.writerow([k] + ['']*7)
            else:
                writer.writerow([k, v['overall_min_of_mins'], v['overall_mean_of_means'], v['overall_max_of_maxs'], v['avg_session_min'], v['avg_session_mean'], v['avg_session_max'], v['num_sessions']])

    print('Wrote', out_json, 'and', out_csv)
    print('\nSummary:')
    for k, v in res['aggregates'].items():
        print(k)
        print(json.dumps(v, indent=2))


if __name__ == '__main__':
    main()
