"""
Analyze Test 1 results.
  python analyze_test1.py                 (uses RESULTS_FOLDER below)
  python analyze_test1.py <other folder>  (overrides it)
Reads every P*_test1.csv in the folder and writes into results/analysis/:
  accuracy_by_command.csv, confusion_counts.csv, confusion_percent.csv,
  response_time_by_command.csv, and three figures (PNG).
"""
import sys, glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMMANDS = ["Forward (W)", "Back (S)", "Left (A)", "Right (D)", "Jump", "Sprint",
            "Look Left", "Look Right", "Look Up", "Look Down"]
DETECTED = COMMANDS + ["No response"]
TARGET_MS = 400
MIN_RT_MS = 200      # faster than human reaction -> movement started before the cue; left out of time stats
RESULTS_FOLDER = r"C:\Users\tranl\OneDrive - Johns Hopkins\Fall 2026\Instrumentation\Bio gaming\results"

# Use a folder from the command line only if it's a real folder (Jupyter adds its own arguments)
folder = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else RESULTS_FOLDER

# Every P*.csv in the folder (any name after P#), except backup copies
files = [p for p in sorted(glob.glob(os.path.join(folder, "P*.csv"))) if "backup" not in os.path.basename(p).lower()]
if not files:
    sys.exit(f"No P*.csv files found in: {folder}")
print("Files loaded:"); [print("  ", os.path.basename(p)) for p in files]
df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
if df.duplicated(["participant", "trial", "attempt", "valid"]).any():
    sys.exit("Same participant appears in two files - keep only one file per participant in the folder.")
out = os.path.join(folder, "analysis"); os.makedirs(out, exist_ok=True)

# Drop trials the operator marked for redo (X): any (participant, trial, attempt) with a valid=0 row
bad = df[df.valid == 0][["participant", "trial", "attempt"]].drop_duplicates()
df = df.merge(bad.assign(_bad=1), how="left", on=["participant", "trial", "attempt"])
df = df[df._bad.isna()].drop(columns="_bad")
print(f"{len(df)} valid trials from {df.participant.nunique()} participants")

# ---- accuracy: per participant per command, then mean ± SD across participants
acc = df.pivot_table(index="cue", columns="participant", values="correct", aggfunc="mean").reindex(COMMANDS)
acc.columns = [f"P{c}" for c in acc.columns]
summ = acc.copy()
summ["mean"], summ["sd"], summ["n"] = acc.mean(axis=1), acc.std(axis=1, ddof=1), acc.count(axis=1)
by_type = df.groupby(["cue_type", "participant"]).correct.mean().unstack()
overall = df.groupby("participant").correct.mean()
for name, s in [("Head commands", by_type.loc["Head"]), ("Eye commands", by_type.loc["Eye"]), ("OVERALL", overall)]:
    row = {f"P{p}": v for p, v in s.items()}
    row.update(mean=s.mean(), sd=s.std(ddof=1), n=s.count())
    summ.loc[name] = pd.Series(row)
(summ * [100 if c not in ("n",) else 1 for c in summ.columns]).round(1).to_csv(os.path.join(out, "accuracy_by_command.csv"))
print("\nAccuracy (%):\n", (summ[["mean", "sd", "n"]] * [100, 100, 1]).round(3))

# ---- confusion matrix (pooled)
cm = pd.crosstab(df.cue, df.detected).reindex(index=COMMANDS, columns=DETECTED, fill_value=0)
extra = [c for c in df.detected.unique() if c not in DETECTED]
cm.to_csv(os.path.join(out, "confusion_counts.csv"))
cmp = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100
cmp.round(1).to_csv(os.path.join(out, "confusion_percent.csv"))

# ---- response time (correct trials only): cue on screen -> device output
ok = df[(df.correct == 1) & df.response_ms.notna()]
fast = ok[ok.response_ms < MIN_RT_MS]
if len(fast):
    print(f"\n{len(fast)} correct trial(s) under {MIN_RT_MS} ms (anticipation - left out of time stats):")
    print(fast[["participant", "trial", "cue", "response_ms"]].to_string(index=False))
    fast.to_csv(os.path.join(out, "anticipatory_trials.csv"), index=False)
ok = ok[ok.response_ms >= MIN_RT_MS]
rt = ok.groupby("cue").response_ms.agg(["count", "median", "mean", "std", "min", "max"]).reindex(COMMANDS)
rt["% under 400 ms"] = ok.groupby("cue").response_ms.apply(lambda s: (s < TARGET_MS).mean() * 100).reindex(COMMANDS)
rt.round(0).to_csv(os.path.join(out, "response_time_by_command.csv"))
head_rt = ok[ok.cue_type == "Head"].groupby("participant").response_ms.median()
eye_rt = ok[ok.cue_type == "Eye"].groupby("participant").response_ms.median()
print(f"\nMedian response time: head {head_rt.mean():.0f} ms, eye {eye_rt.mean():.0f} ms (mean of per-person medians)")
try:
    from scipy.stats import wilcoxon
    paired = pd.concat([head_rt, eye_rt], axis=1).dropna()
    if len(paired) >= 5:
        print("Wilcoxon signed-rank head vs eye (per-person medians): p =", round(wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue, 4))
except ImportError:
    pass

# ---- figures
# Figure 1 - mean and SD
colors = ["#2F5597" if c in COMMANDS[:6] else "#C55A11" for c in COMMANDS]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(COMMANDS, summ.loc[COMMANDS, "mean"] * 100, yerr=summ.loc[COMMANDS, "sd"].fillna(0) * 100,
       capsize=4, color=colors)
ax.set_ylabel("Accuracy (%, mean ± SD)")
ax.set_ylim(0, 110)
ax.set_title(f"Command Recognition Accuracy by Input Modality (n = {df.participant.nunique()})")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2F5597", label="Head (IMU)"), Patch(color="#C55A11", label="Eye (camera)")], loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_accuracy.png"), dpi=200); plt.close(fig)


# Figure 2 - Confusion matrix (% of cued trials, pooled)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cmp.values, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(DETECTED))); ax.set_xticklabels(DETECTED, rotation=45, ha="right")
ax.set_yticks(range(len(COMMANDS))); ax.set_yticklabels(COMMANDS)
for i in range(cmp.shape[0]):
    for j in range(cmp.shape[1]):
        v = cmp.values[i, j]
        if v > 0: ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 60 else "black", fontsize=8)
ax.set_xlabel("Detected command"); ax.set_ylabel("Cued command");
ax.set_title("Classification Confusion Matrix: Cued vs. Detected Commands")
fig.colorbar(im, ax=ax, fraction=0.04, label="%")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig_confusion.png"), dpi=200); plt.close(fig)


# Figure 3 - Response time per command
fig, ax = plt.subplots(figsize=(9, 4.5))
data = [ok[ok.cue == c].response_ms.values for c in COMMANDS]
bp = ax.boxplot(data, patch_artist=True, showfliers=True)
for p, c in zip(bp["boxes"], colors): p.set_facecolor(c); p.set_alpha(0.6)
ax.set_xticks(range(1, len(COMMANDS) + 1)); ax.set_xticklabels(COMMANDS, rotation=35, ha="right")
ax.set_ylabel("Latency (ms)")
ax.set_title("System Response Latency by Command (Cue to Output)")
fig.tight_layout()


fig.savefig(os.path.join(out, "fig_response_time.png"), dpi=200); 
plt.close(fig)
if extra: print("Note: unexpected detected labels:", extra)
print("\nSaved tables and figures to", out)
