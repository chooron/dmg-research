#!/usr/bin/env python3
import glob, json, os, sys, numpy as np, pandas as pd

def audit_convergence():
    with open('/autodl-fs/data/531sub_id.txt') as f:
        text = f.read().strip()
        try:
            all_531 = [str(x).zfill(8) for x in json.loads(text)]
        except Exception:
            all_531 = [x.strip().zfill(8) for x in text.splitlines() if x.strip()]

    models = ['XAJ', 'XAJ_CN', 'XAJ_TGD']

    for model_key in models:
        out_dir = f'/root/outputs/full_model_series_calibration/v1/tasks/{model_key}'
        result_files = glob.glob(os.path.join(out_dir, '**/seed_101/**/result.json'), recursive=True)
        print(f'=== Auditing {model_key}: Found {len(result_files)} result files ===')

        basin_records = {b: {} for b in all_531}

        for rf in result_files:
            try:
                d = json.load(open(rf))
                b_id = str(d.get('basin_id')).zfill(8)
                s_id = d.get('start_idx', d.get('start_id'))
                tr_kge = d.get('best_train_kge')
                theta = d.get('best_theta_normalized')
                tot_gen = d.get('total_generations', 300)

                # Check trace.json
                trace_file = os.path.join(os.path.dirname(rf), 'trace.json')
                plateau_gen = None
                if os.path.exists(trace_file):
                    try:
                        t_data = json.load(open(trace_file))
                        best_fits = [x['best_fitness'] for x in t_data]
                        final_fit = best_fits[-1]
                        # Find generation where best_fitness gets within 1e-4 of final fitness
                        for g_idx, f_val in enumerate(best_fits):
                            if (final_fit - f_val) < 1e-4:
                                plateau_gen = g_idx + 1
                                break
                    except Exception:
                        pass

                if b_id in basin_records and s_id in [0, 1, 2]:
                    basin_records[b_id][s_id] = {
                        'train_kge': tr_kge,
                        'theta': theta,
                        'total_gen': tot_gen,
                        'plateau_gen': plateau_gen
                    }
            except Exception as e:
                pass

        summary_rows = []
        for b_id in all_531:
            starts = basin_records[b_id]
            valid_s = [s for s in [0, 1, 2] if s in starts]
            if len(valid_s) == 3:
                kges = [starts[s]['train_kge'] for s in [0, 1, 2]]
                thetas = [np.array(starts[s]['theta']) for s in [0, 1, 2]]

                kge_spread = max(kges) - min(kges)

                p_dists = []
                for i in range(3):
                    for j in range(i+1, 3):
                        p_dists.append(np.linalg.norm(thetas[i] - thetas[j]))
                param_dist_mean = float(np.mean(p_dists))

                plateaus = [starts[s]['plateau_gen'] for s in [0, 1, 2] if starts[s]['plateau_gen'] is not None]
                avg_plateau = float(np.mean(plateaus)) if len(plateaus) > 0 else np.nan

                summary_rows.append({
                    'basin_id': b_id,
                    'kge_spread': kge_spread,
                    'param_dist_mean': param_dist_mean,
                    'avg_plateau_gen': avg_plateau,
                    'kge_max': max(kges),
                    'kge_min': min(kges),
                    'kge_med': float(np.median(kges))
                })

        df_sum = pd.DataFrame(summary_rows)
        out_csv = f'/root/outputs/full_model_series_calibration/v1/{model_key}_convergence_audit.csv'
        df_sum.to_csv(out_csv, index=False)
        print(f'Saved {len(df_sum)} basin convergence audit records for {model_key} to {out_csv}')

        q75 = df_sum['kge_spread'].quantile(0.75)
        q25 = df_sum['kge_spread'].quantile(0.25)
        print(f'-- {model_key} Overall Convergence Audit ({len(df_sum)} basins) --')
        print(f'   Start KGE Spread Med: {df_sum["kge_spread"].median():.6f} | IQR: {q75 - q25:.6f} | Max: {df_sum["kge_spread"].max():.6f}')
        print(f'   Basins with Start KGE Spread < 0.01: {(df_sum["kge_spread"] < 0.01).sum()} / {len(df_sum)} ({(df_sum["kge_spread"] < 0.01).mean()*100:.1f}%)')
        print(f'   Basins with Start KGE Spread < 0.001: {(df_sum["kge_spread"] < 0.001).sum()} / {len(df_sum)} ({(df_sum["kge_spread"] < 0.001).mean()*100:.1f}%)')
        print(f'   Mean Param Vector Distance Med: {df_sum["param_dist_mean"].median():.4f}')
        print(f'   Avg Plateau Generation Med: {df_sum["avg_plateau_gen"].median():.1f} / 300 generations\n')

if __name__ == '__main__':
    audit_convergence()
