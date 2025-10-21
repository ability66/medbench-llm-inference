import os, json, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS_CSV = "result/metrics_promptcblue.csv"
df = pd.read_csv(METRICS_CSV)

# 排序
df = df.sort_values("rouge-l", ascending=False).reset_index(drop=True)

# 1) 排行榜：ROUGE-L
plt.figure()
plt.bar(df["model"], df["rouge-l"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROUGE-L (F1)")
plt.title("Model Leaderboard (ROUGE-L)")
plt.tight_layout()
plt.savefig("result/fig_leaderboard_rougel.png", dpi=200)

# 1b) BLEU（单独一张）
plt.figure()
plt.bar(df["model"], df["bleu"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("BLEU")
plt.title("Model Leaderboard (BLEU)")
plt.tight_layout()
plt.savefig("result/fig_leaderboard_bleu.png", dpi=200)

# 2) 长度—质量散点
plt.figure()
plt.scatter(df["avg_chars"], df["rouge-l"])
for _, r in df.iterrows():
    plt.annotate(r["model"], (r["avg_chars"], r["rouge-l"]), xytext=(3,3), textcoords="offset points")
# 拟合一条简单的线看趋势
if len(df) >= 2:
    m, b = np.polyfit(df["avg_chars"], df["rouge-l"], 1)
    xs = np.linspace(df["avg_chars"].min(), df["avg_chars"].max(), 100)
    plt.plot(xs, m*xs + b)
plt.xlabel("Average Characters of Output")
plt.ylabel("ROUGE-L (F1)")
plt.title("Length vs Quality")
plt.tight_layout()
plt.savefig("result/fig_len_vs_quality.png", dpi=200)

