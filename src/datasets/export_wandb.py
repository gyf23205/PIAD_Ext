import pandas as pd
import wandb

api = wandb.Api()

# --- Configuration ---
# SWEEP_IDS = ["6g5mb1nv", "etggoi4e", "vsxqrb36", "4jfgip4f", "xzo9n5op", "5j2g8ith", "q5okhzkh", "8ax0v9uv"]   # add your sweep IDs here
SWEEP_IDS = ["grilepd2"]
COLUMNS = ["dataset", "seed", "ratios", "bin_acc", "bin_f1", "bin_recall", "f1_macro", "f1_weighted", "test_auc", "mh_acc", "mh_recall", "f_aff", "p_aff", "r_aff", "mcc"]
# ---------------------

rows = []
for sweep_id in SWEEP_IDS:
    sweep = api.sweep(f"guoyifan489/PIAD_Ext/{sweep_id}")
    for run in sweep.runs:
        row = {"name": run.name, "sweep": sweep_id}
        for col in COLUMNS:
            if col.startswith("config."):
                key = col[len("config."):]
                row[col] = run.config.get(key)
            elif col.startswith("summary."):
                key = col[len("summary."):]
                row[col] = run.summary.get(key)
            else:
                # Bare key: look in config first, then summary
                if col in run.config:
                    row[col] = run.config[col]
                else:
                    row[col] = run.summary.get(col)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("resweep_ALFA.csv", index=False)
