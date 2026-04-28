import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("reports/tables/phase5a_gnn_metrics.csv")

metrics = ["r2", "mae", "spearman"]

for m in metrics:
    plt.figure()
    plt.bar(df["model"], df[m])
    plt.title(f"GNN Comparison — {m.upper()}")
    plt.ylabel(m.upper())
    plt.xticks(rotation=20)
    plt.tight_layout()

    path = Path(f"reports/figures/p5a_metric_{m}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

print("✓ Phase 5A figures saved")