import os, json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS_CSV = "result/metrics_promptcblue.csv"
df = pd.read_csv(METRICS_CSV)

# 排序
df = df.sort_values("ROUGE-L", ascending=False).reset_index(drop=True)

# 1) 排行榜：ROUGE-L
plt.figure()
plt.bar(df["model"], df["ROUGE-L"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROUGE-L (F1)")
plt.title("Model Leaderboard (ROUGE-L)")
plt.tight_layout()
plt.savefig("result/fig_leaderboard_rougel.png", dpi=200)

# 1b) BLEU（单独一张）
plt.figure()
plt.bar(df["model"], df["BLEU"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("BLEU")
plt.title("Model Leaderboard (BLEU)")
plt.tight_layout()
plt.savefig("result/fig_leaderboard_bleu.png", dpi=200)

# 2) 长度—质量散点
plt.figure()
plt.scatter(df["avg_chars"], df["ROUGE-L"])
for _, r in df.iterrows():
    plt.annotate(r["model"], (r["avg_chars"], r["ROUGE-L"]), xytext=(3,3), textcoords="offset points")
# 拟合一条简单的线看趋势
if len(df) >= 2:
    m, b = np.polyfit(df["avg_chars"], df["ROUGE-L"], 1)
    xs = np.linspace(df["avg_chars"].min(), df["avg_chars"].max(), 100)
    plt.plot(xs, m*xs + b)
plt.xlabel("Average Characters of Output")
plt.ylabel("ROUGE-L (F1)")
plt.title("Length vs Quality")
plt.tight_layout()
plt.savefig("result/fig_len_vs_quality.png", dpi=200)

# 3) 分节热力图（若有 secRL_* 列）
sec_cols = [c for c in df.columns if c.startswith("secRL_")]
if sec_cols:
    mat = df[sec_cols].to_numpy()
    plt.figure()
    im = plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(sec_cols)), [c.replace("secRL_","") for c in sec_cols], rotation=45, ha="right")
    plt.yticks(range(len(df)), df["model"])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Section-wise ROUGE-L (F1)")
    plt.tight_layout()
    plt.savefig("result/fig_section_heatmap.png", dpi=200)

print("Saved figures under ./result/")
