#!/usr/bin/env python3
import glob, json, os, sys, torch, pandas as pd, numpy as np
from ablation.ic_core.data_adapter import load_531_bundle
from ablation.ic_core.runtime import ICObjectiveRuntime

def evaluate_all_starts():
    with open('/autodl-fs/data/531sub_id.txt') as f:
        text = f.read().strip()
        try:
            all_531 = [str(x).zfill(8) for x in json.loads(text)]
        except Exception:
            all_531 = [x.strip().zfill(8) for x in text.splitlines() if x.strip()]

    cfg_path = 'ablation/configs/ic_foundation_531_v1.json'
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['basin_list_path'] = '/autodl-fs/data/531sub_id.txt'
    cfg['gage_ids_path'] = '/autodl-fs/data/gage_id.npy'
    cfg['dates_path'] = '/autodl-fs/data/camels_dates.npy'
    cfg['dataset_path'] = '/autodl-fs/data/camels_dataset'
    cfg['periods'] = {
        'warmup': {'start': '1980-10-01', 'end': '1981-09-30'},
        'train': {'start': '1981-10-01', 'end': '1995-09-30'},
        'test': {'start': '1995-10-01', 'end': '2010-09-30'}
    }

    bundle = load_531_bundle(cfg)

    for model_key in ['XAJ', 'XAJ_CN', 'XAJ_TGD']:
        print(f'=== Processing {model_key} All Starts (seed_101) ===')
        runtime = ICObjectiveRuntime(bundle=bundle, config=cfg, model_key=model_key)
        out_dir = f'/root/outputs/full_model_series_calibration/v1/tasks/{model_key}'

        start_records = {s: {} for s in [0, 1, 2]}
        files = glob.glob(os.path.join(out_dir, f'**/seed_101/**/result.json'), recursive=True)
        print(f'Found {len(files)} result files for {model_key} seed_101.')

        for f in files:
            try:
                d = json.load(open(f))
                b_id = str(d.get('basin_id')).zfill(8)
                s_id = d.get('start_idx', d.get('start_id'))
                tr_kge = d.get('best_train_kge')
                theta = d.get('best_theta_normalized')
                if b_id in all_531 and s_id in [0, 1, 2] and tr_kge is not None and theta is not None:
                    start_records[s_id][b_id] = {
                        'train_kge': tr_kge,
                        'theta': theta
                    }
            except Exception:
                pass

        # Evaluate each start
        start_eval_results = {}
        for s_id in [0, 1, 2]:
            rec = start_records[s_id]
            valid_bids = [b for b in all_531 if b in rec]
            print(f'Start {s_id}: {len(valid_bids)} / 531 valid basin results.')

            b_indices = [runtime.bundle.basin_ids.index(b) for b in valid_bids]
            thetas = [rec[b]['theta'] for b in valid_bids]
            theta_matrix = torch.tensor(thetas, dtype=torch.float32, device=cfg['device']).unsqueeze(1)

            eval_res = runtime.evaluate_candidates(theta_matrix, basin_indices=b_indices, split='test')
            test_kges = np.asarray(eval_res.fitness).flatten()

            start_eval_results[s_id] = {}
            for b, kge_val in zip(valid_bids, test_kges):
                start_eval_results[s_id][b] = {
                    'train_kge': rec[b]['train_kge'],
                    'test_kge': float(kge_val)
                }

        # Calculate per-basin Median-of-3 and Best-of-3
        per_basin_rows = []
        for b_id in all_531:
            tr_vals = [start_eval_results[s][b_id]['train_kge'] for s in [0, 1, 2] if b_id in start_eval_results[s]]
            te_vals = [start_eval_results[s][b_id]['test_kge'] for s in [0, 1, 2] if b_id in start_eval_results[s]]

            if len(tr_vals) > 0:
                med_tr = float(np.median(tr_vals))
                med_te = float(np.median(te_vals))
                best_idx = int(np.argmax(tr_vals))
                best_tr = float(tr_vals[best_idx])
                best_te = float(te_vals[best_idx])
            else:
                med_tr, med_te, best_tr, best_te = np.nan, np.nan, np.nan, np.nan

            per_basin_rows.append({
                'basin_id': b_id,
                'median_train_kge': round(med_tr, 4),
                'median_test_kge': round(med_te, 4),
                'best_train_kge': round(best_tr, 4),
                'best_test_kge': round(best_te, 4),
                'num_starts': len(tr_vals)
            })

        df_out = pd.DataFrame(per_basin_rows)
        out_csv = f'/root/outputs/full_model_series_calibration/v1/{model_key}_ic_median_and_best_of_3.csv'
        df_out.to_csv(out_csv, index=False)
        print(f'Saved {model_key} IC summary to {out_csv}')

        valid_df = df_out.dropna(subset=['median_train_kge'])
        print(f'-- {model_key} Median-of-3 Starts Metrics ({len(valid_df)} basins) --')
        print(f'   Train KGE - Median: {valid_df["median_train_kge"].median():.4f} | Mean: {valid_df["median_train_kge"].mean():.4f}')
        print(f'   Test  KGE - Median: {valid_df["median_test_kge"].median():.4f} | Mean: {valid_df["median_test_kge"].mean():.4f}\n')

if __name__ == '__main__':
    evaluate_all_starts()
