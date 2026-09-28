"""
EMG Full Analysis Pipeline – Subject 01  (SIT-TO-STAND)
Conditions: NoExo | NoTorque | NN_S2S_10nm

Processing pipeline per muscle:
  1. Bandpass filter  (4th-order Butterworth, 50–450 Hz)
  2. Demean
  3. Full-wave rectify
  4. Lowpass envelope  (6th-order Butterworth, 5 Hz)
  5. MVC normalisation (global max across MVC trials + trial peak)

Segmentation strategy:
  All three conditions use EMG-based automatic segmentation, retuned for
  the sit-to-stand kinematic signature (quiet → rise → sustained plateau
  → fall → quiet).  Cycles are detected by pairing the lowering and rising movement bursts,
  keeping the quiet seated hold inside each complete cycle.

EMG-based segmentation algorithm (paired movement events):
  A composite multi-muscle signal (sum of envelopes for the primary
  knee extensors: RF_R, RF_L, VL_R, VL_L, VM_R, VM_L) is computed,
  z-score normalised and Gaussian smoothed.  Two levels are set on the
  composite dynamic range: an ONSET level marking the rising/falling
  edges of each activation bout (= cycle start/end at quiet baseline),
  and a higher PLATEAU level that a genuine s2s bout must exceed inside.
  Contiguous regions above the onset level define candidate cycles;
  near bouts separated by a short quiet gap are merged, and bouts are
  kept only if they plateau above the plateau level and fall within
  the allowed duration window.

Outputs per condition:
  Fig 6 – Full EMG timeseries with cycle-boundary overlay
  Fig 5 – Individual s2s cycles overlaid with mean
  Fig 4 – Cycle-averaged mean ± SD
  CSV   – EMG cycle-average data
  CSV   – EMG-based segmentation times (start / end, seconds)

APDM / P90:
  After normalisation, the Amplitude Probability Distribution Method is
  applied to every muscle: P90 (90th percentile of all EMG values within
  the valid s2s cycles) is reported and used in the CSV output.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# ─── Configuration ─────────────────────────────────────────────────────────────
BP_LOW, BP_HIGH = 50, 450          # bandpass cutoffs (Hz)
BP_ORDER        = 4
LP_CUTOFF       = 5                # envelope lowpass (Hz)
LP_ORDER        = 6
N_POINTS        = 101              # 0–100% s2s cycle grid
MIN_CYCLE_DUR   = 1.5              # s — discard cycles shorter than this
MAX_CYCLE_DUR   = 8.0              # s — discard cycles longer than this (merge guard)
MIN_CYCLE_GAP   = 1.0              # s — minimum quiet gap between bouts
COMPOSITE_SMOOTH_SEC = 0.4         # Gaussian smoothing window for composite signal (s)
ONSET_PERCENTILE     = 25          # % of composite range: bout on/off (edge) threshold
PLATEAU_PERCENTILE   = 55          # % of composite range: must exceed inside a bout
APDM_PERCENTILE      = 90          # P90

# ─── Paths ────────────────────────────────────────────────────────────────────
UPLOAD_DIR = r"D:\Saanya Dell XPS 9500\VSCode\BioDynamics\s2s"
MVC_DIR    = r"D:\Saanya Dell XPS 9500\VSCode\BioDynamics\s2s"
OUT_DIR    = r"D:\Saanya Dell XPS 9500\VSCode\BioDynamics\s2s\output"

MVC_PLOT_DIR = os.path.join(OUT_DIR, "MVC_Plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MVC_PLOT_DIR, exist_ok=True)

# ─── Muscle & plot layout ──────────────────────────────────────────────────────
MUSCLES = [
    "GMax_R", "GMax_L", "BF_R",  "BF_L",
    "ST_R",   "ST_L",   "RF_R",  "RF_L",
    "VL_R",   "VL_L",   "VM_R",  "VM_L",
    "IL_R",   "IL_L",   "TFL_R", "TFL_L",
]

# 4×4 grid layout: alphabetical, Left then Right
PLOT_ORDER = [
    "BF_L",  "BF_R",  "GMax_L", "GMax_R",
    "IL_L",  "IL_R",  "RF_L",   "RF_R",
    "ST_L",  "ST_R",  "TFL_L",  "TFL_R",
    "VL_L",  "VL_R",  "VM_L",   "VM_R",
]

# Muscles used to build the composite signal for EMG-based segmentation.
# These are the primary knee extensors most active throughout the s2s.
COMPOSITE_MUSCLES = ["RF_R", "RF_L", "VL_R", "VL_L", "VM_R", "VM_L"]

# ─── MVC files ────────────────────────────────────────────────────────────────
MVC_FILES = [
    ("EMG_Recording_Subject01_HipExt_L_MVC.csv",   ["GMax_L"]),
    ("EMG_Recording_Subject01_HipExt_R_MVC.csv",   ["GMax_R"]),
    ("EMG_Recording_Subject01_HipFlex_L_MVC.csv",  ["IL_L", "TFL_L"]),
    ("EMG_Recording_Subject01_HipFlex_R_MVC.csv",  ["IL_R", "TFL_R"]),
    ("EMG_Recording_Subject01_KneeExt_L_MVC.csv",  ["RF_L", "VL_L", "VM_L"]),
    ("EMG_Recording_Subject01_KneeExt_R_MVC.csv",  ["RF_R", "VL_R", "VM_R"]),
    ("EMG_Recording_Subject01_KneeFlex_L_MVC.csv", ["BF_L", "ST_L"]),
    ("EMG_Recording_Subject01_KneeFlex_R_MVC.csv", ["BF_R", "ST_R"]),
]

# Condition definitions:
#   (label, emg_filename, segmentation_mode, seg_slice_or_None, file_tag)
# All conditions now use EMG-based automatic segmentation.
TRIALS = [
    ("No Exo",    "EMG_Recording_Subject01_NoExo_s2s.csv",
     "emg",    None,     "NoExo_s2s"),
    ("No Torque", "EMG_Recording_Subject01_NoTorque_s2s.csv",
     "emg",    None,     "NoTorque_s2s"),
    ("NN Torque",   "EMG_Recording_Subject01_NN_s2s_10nm.csv",
     "emg",    None,     "NN_s2s"),
]

# ─── Plot colours ──────────────────────────────────────────────────────────────
SEG_ENV_COLOR   = "#E89537"
SEG_START_COLOR = "blue"
SEG_END_COLOR   = "red"
INDIV_COLOR     = "#042F53"
MEAN_COLOR      = "#030303"
CYCLE_LINE      = "#1C6BB0"
CYCLE_FILL      = "#AEC8E8"
MVC_TARGET_COLOR  = "#C0392B"
MVC_OTHER_COLOR   = "#7F8C8D"
MVC_PEAK_COLOR    = "#16A085"


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def get_fs(df):
    """Effective sampling rate from timestamps (ms units)."""
    t = (df["timestamp"].values - df["timestamp"].values[0]) / 1000.0
    return len(df) / t[-1]


def process_emg(raw, fs):
    """Bandpass → demean → full-wave rectify → lowpass envelope → baseline correct."""
    nyq   = 0.5 * fs
    b, a  = butter(BP_ORDER, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    bp    = filtfilt(b, a, raw)
    dm    = bp - np.mean(bp)
    rect  = np.abs(dm)
    b2, a2 = butter(LP_ORDER, LP_CUTOFF / nyq, btype="low")
    env   = filtfilt(b2, a2, rect)

    # Baseline correction: subtract the resting-level floor so that
    # periods with no activation sit at ~0.  The 5th percentile of the
    # envelope is a robust estimate of the noise floor / resting baseline.
    baseline = np.percentile(env, 5)
    env = np.maximum(env - baseline, 0.0)

    return env


def segment_cycles_emg(time_s, envelopes, fs,
                       composite_muscles=COMPOSITE_MUSCLES,
                       smooth_sec=COMPOSITE_SMOOTH_SEC,
                       onset_pct=ONSET_PERCENTILE,
                       plateau_pct=PLATEAU_PERCENTILE,
                       min_cycle_dur=MIN_CYCLE_DUR,
                       max_cycle_dur=MAX_CYCLE_DUR,
                       min_cycle_gap=MIN_CYCLE_GAP):
    """Detect complete standing -> sitting -> seated hold -> standing cycles.

    S2S differs from a squat because one complete repetition normally contains
    TWO movement-related EMG bursts separated by the deliberately quiet seated
    hold. Therefore the quiet interval is retained inside the cycle rather than
    being treated as a cycle boundary.

    The detector:
      1. forms a normalized, smoothed knee-extensor composite EMG signal;
      2. finds meaningful movement peaks;
      3. estimates the shorter, within-cycle peak spacing from the recording;
      4. pairs adjacent peaks whose spacing matches that within-cycle spacing;
      5. places cycle boundaries at low-activation valleys outside each pair.

    This is the EMG fallback. Xsens joint-angle segmentation is preferable when
    synchronized Xsens files are available.
    """
    comp = np.zeros(len(time_s), dtype=float)
    used = []
    for m in composite_muscles:
        if m in envelopes:
            comp += np.asarray(envelopes[m], dtype=float)
            used.append(m)
    if not used or np.allclose(comp.std(), 0):
        print("    WARNING: composite EMG is empty/flat.")
        return pd.DataFrame(columns=["Start_Time_Sec", "End_Time_Sec",
                                     "SitPeak_Time_Sec", "StandPeak_Time_Sec"])

    comp_z = (comp - np.mean(comp)) / (np.std(comp) + 1e-12)
    comp_smooth = gaussian_filter1d(comp_z, sigma=max(1, smooth_sec * fs))

    lo, hi = np.percentile(comp_smooth, [1, 99])
    rng = max(hi - lo, 1e-12)
    peak_level = lo + (plateau_pct / 100.0) * rng
    baseline_level = lo + (onset_pct / 100.0) * rng

    # Peaks are movement events, not repetitions. A complete S2S rep should
    # contain a lowering event followed by a rising event.
    peaks, _ = find_peaks(
        comp_smooth,
        height=peak_level,
        prominence=0.12 * rng,
        distance=max(1, int(1.0 * fs)),
    )
    if len(peaks) < 2:
        print("    WARNING: fewer than two movement peaks found.")
        return pd.DataFrame(columns=["Start_Time_Sec", "End_Time_Sec",
                                     "SitPeak_Time_Sec", "StandPeak_Time_Sec"])

    peak_times = time_s[peaks]
    gaps = np.diff(peak_times)

    # In metronome-guided S2S, within-cycle sit->stand spacing is normally the
    # shorter member of the alternating gap pattern. Estimate it robustly from
    # the lower half of adjacent-peak gaps instead of assuming peak parity.
    finite_gaps = gaps[np.isfinite(gaps)]
    if len(finite_gaps) == 0:
        return pd.DataFrame(columns=["Start_Time_Sec", "End_Time_Sec",
                                     "SitPeak_Time_Sec", "StandPeak_Time_Sec"])
    gap_mid = float(np.median(finite_gaps))
    short_gaps = finite_gaps[finite_gaps <= gap_mid]
    typical_within = float(np.median(short_gaps)) if len(short_gaps) else gap_mid
    pair_max_gap = min(8.0, max(3.0, typical_within * 1.35))
    pair_min_gap = max(1.5, typical_within * 0.55)

    # Greedy pairing by timing, not by even/odd peak number. This prevents a
    # partial movement at the beginning of a recording from shifting all pairs.
    pairs = []
    i = 0
    while i < len(peaks) - 1:
        gap = peak_times[i + 1] - peak_times[i]
        if pair_min_gap <= gap <= pair_max_gap:
            pairs.append((peaks[i], peaks[i + 1]))
            i += 2
        else:
            i += 1

    rows = []
    last_end = -np.inf
    for p1, p2 in pairs:
        # Boundary before the lowering burst: minimum composite activation in
        # the interval after the previous paired event and before p1.
        prev_candidates = peaks[peaks < p1]
        left_limit = int(prev_candidates[-1]) if len(prev_candidates) else 0
        left_seg = comp_smooth[left_limit:p1 + 1]
        s_idx = left_limit + int(np.argmin(left_seg)) if len(left_seg) else p1

        # Boundary after rising burst: minimum activation before the next event.
        next_candidates = peaks[peaks > p2]
        right_limit = int(next_candidates[0]) if len(next_candidates) else len(time_s) - 1
        right_seg = comp_smooth[p2:right_limit + 1]
        e_idx = p2 + int(np.argmin(right_seg)) if len(right_seg) else p2

        # If minima remain above the onset level, walk outward to the nearest
        # sample at/below onset where possible.
        before = np.where(comp_smooth[:p1] <= baseline_level)[0]
        if len(before):
            cand = before[-1]
            if cand >= left_limit:
                s_idx = cand
        after = np.where(comp_smooth[p2:right_limit + 1] <= baseline_level)[0]
        if len(after):
            e_idx = p2 + after[0]

        t_start, t_end = float(time_s[s_idx]), float(time_s[e_idx])
        dur = t_end - t_start
        # Full S2S cycles are longer than a single burst. Keep broad limits so
        # the metronome hold is not accidentally rejected.
        if max(3.0, min_cycle_dur) <= dur <= max(12.0, max_cycle_dur):
            if t_start > last_end - 0.05:
                rows.append({
                    "Start_Time_Sec": round(t_start, 4),
                    "End_Time_Sec": round(t_end, 4),
                    "SitPeak_Time_Sec": round(float(time_s[p1]), 4),
                    "StandPeak_Time_Sec": round(float(time_s[p2]), 4),
                })
                last_end = t_end

    print(f"    Movement peaks: {len(peaks)}")
    print(f"    Typical within-cycle peak gap: {typical_within:.2f} s")
    print(f"    Accepted peak-pair gap: {pair_min_gap:.2f}–{pair_max_gap:.2f} s")
    print(f"    Full S2S cycles retained: {len(rows)}")
    return pd.DataFrame(rows, columns=["Start_Time_Sec", "End_Time_Sec",
                                      "SitPeak_Time_Sec", "StandPeak_Time_Sec"])


def compute_apdm_p90(cycles_arr, percentile=APDM_PERCENTILE):
    """Return P90 (or requested percentile) across all normalized cycle samples."""
    vals = np.asarray(cycles_arr, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, percentile)) if vals.size else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 1. Process MVC trials – accumulate per-muscle maximum
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Processing MVC trials …")
print("=" * 65)

mvc_global       = {m: [] for m in MUSCLES}
mvc_per_trial    = {}

for mvc_fname, target_muscles in MVC_FILES:
    trial_tag = os.path.splitext(mvc_fname)[0].replace(
        "EMG_Recording_Subject01_", "")
    print(f"  {trial_tag}: target = {target_muscles}")

    df = pd.read_csv(os.path.join(MVC_DIR, mvc_fname))
    fs = get_fs(df)
    t  = (df["timestamp"].values - df["timestamp"].values[0]) / 1000.0

    envelopes = {}
    peaks     = {}
    peak_idxs = {}
    for m in MUSCLES:
        env = process_emg(df[m].values.astype(float), fs)
        envelopes[m] = env
        peaks[m]     = float(env.max())
        peak_idxs[m] = int(np.argmax(env))
        mvc_global[m].append(peaks[m])
    mvc_per_trial[trial_tag] = peaks

    DISPLAY_STEP = max(1, int(round(fs / 50)))   # downsample to ~50 Hz for plotting

    # ── 16-muscle 4×4 grid plot for this MVC trial ──
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
    fig.suptitle(
        f"Subject01 — MVC trial: {trial_tag}\n"
        f"target muscles highlighted in red  |  "
        f"green dot = peak (used as MVC for that muscle)",
        fontsize=12, fontweight="bold", y=1.005,
    )

    for ax_idx, muscle in enumerate(PLOT_ORDER):
        ax        = axes[ax_idx // 4][ax_idx % 4]
        is_target = muscle in target_muscles
        col   = MVC_TARGET_COLOR if is_target else MVC_OTHER_COLOR
        lw    = 0.9  if is_target else 0.55
        alpha = 0.95 if is_target else 0.70

        env_ds = envelopes[muscle][::DISPLAY_STEP]
        t_ds   = t[::DISPLAY_STEP]
        ax.plot(t_ds, env_ds, color=col, linewidth=lw, alpha=alpha)

        pk_idx = peak_idxs[muscle]
        ax.plot(t[pk_idx], envelopes[muscle][pk_idx], "o",
                color=MVC_PEAK_COLOR, markersize=5, zorder=3)

        peak_val      = peaks[muscle]
        title_suffix  = "★" if is_target else ""
        ax.set_title(f"{muscle}{title_suffix}   peak={peak_val:.2e}",
                     fontsize=9,
                     fontweight="bold" if is_target else "normal",
                     pad=3, color=col if is_target else "black")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if ax_idx % 4 == 0:
            ax.set_ylabel("Envelope (V)", fontsize=8)
        if ax_idx // 4 == 3:
            ax.set_xlabel("Time (s)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 1])
    mvc_plot_path = os.path.join(MVC_PLOT_DIR, f"MVC_{trial_tag}.png")
    plt.savefig(mvc_plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Plot saved → {mvc_plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Loop over conditions
# ══════════════════════════════════════════════════════════════════════════════
pct_axis = np.linspace(0, 100, N_POINTS)
all_stats = {}       # accumulate per-condition stats for cross-condition plot

for (cond_label, s2s_fname, seg_mode, seg_slice, file_tag) in TRIALS:
    print(f"\n{'=' * 65}")
    print(f"Condition : {cond_label}  |  segmentation mode: {seg_mode}")
    print(f"{'=' * 65}")

    # ── 2a. Load & process EMG ────────────────────────────────────────────────
    s2s_df = pd.read_csv(os.path.join(UPLOAD_DIR, s2s_fname))
    fs       = get_fs(s2s_df)
    time_s   = (s2s_df["timestamp"].values - s2s_df["timestamp"].values[0]) / 1000.0
    emg_dur  = time_s[-1]
    print(f"  fs = {fs:.1f} Hz  |  duration = {emg_dur:.1f} s")

    s2s_env = {m: process_emg(s2s_df[m].values.astype(float), fs)
                 for m in MUSCLES}

    # MVC = global max across MVC trials AND this trial's peak
    mvc = {m: max(mvc_global[m] + [float(np.max(s2s_env[m]))]) for m in MUSCLES}

    # MVC-normalise (clip at 0 from below; SD bands may naturally exceed 1)
    s2s_norm = {m: np.clip(s2s_env[m] / mvc[m], 0, None) for m in MUSCLES}

    # ── 2b. Segmentation ──────────────────────────────────────────────────────
    # Pair lowering + rising EMG events into one complete S2S cycle
    print("  Running EMG-based paired-event segmentation (sit + stand = one cycle) …")
    cond_seg = segment_cycles_emg(
        time_s, s2s_norm, fs,
        composite_muscles=COMPOSITE_MUSCLES,
        smooth_sec=COMPOSITE_SMOOTH_SEC,
        onset_pct=ONSET_PERCENTILE,
        plateau_pct=PLATEAU_PERCENTILE,
        min_cycle_dur=MIN_CYCLE_DUR,
        max_cycle_dur=MAX_CYCLE_DUR,
        min_cycle_gap=MIN_CYCLE_GAP,
    )
    print(f"  EMG segmentation found: {len(cond_seg)} valid cycles.")

    # Save EMG-based segmentation file
    seg_save_path = os.path.join(
        OUT_DIR, f"EMG_Segmentation_{file_tag}.csv")
    cond_seg.to_csv(seg_save_path, index=False)
    print(f"  ✓ EMG segmentation saved → {seg_save_path}")

    n_cycles = len(cond_seg)
    if n_cycles == 0:
        print("  !! No valid cycles found — skipping condition.")
        continue

    # ── 2c. Time-normalise each cycle (0–100 %) ───────────────────────────────
    cycles = {m: [] for m in MUSCLES}
    for _, row in cond_seg.iterrows():
        t_start, t_end = row["Start_Time_Sec"], row["End_Time_Sec"]
        idx = np.where((time_s >= t_start) & (time_s <= t_end))[0]
        if len(idx) < 20:
            continue
        t_norm = np.linspace(0, 100, len(idx))
        for m in MUSCLES:
            cycles[m].append(np.interp(pct_axis, t_norm, s2s_norm[m][idx]))

    cycles_arr = {m: np.array(cycles[m]) for m in MUSCLES}
    # Update n_cycles in case a few were skipped (< 20 samples)
    n_cycles = len(list(cycles_arr.values())[0])

    # # ── 2c½. Correlation-based artifact rejection ─────────────────────────────
    # # For each cycle, correlate it with the median cycle shape across ALL
    # # muscles, then average those correlations into one quality score.
    # # Cycles below CYCLE_CORR_THRESHOLD are artifacts / outliers.
    # if n_cycles > 3:     # need at least a few to compute a meaningful median
    #     median_cycles = {m: np.median(cycles_arr[m], axis=0) for m in MUSCLES}
    #     quality = np.zeros(n_cycles)
    #     for i in range(n_cycles):
    #         corrs = []
    #         for m in MUSCLES:
    #             r = np.corrcoef(cycles_arr[m][i], median_cycles[m])[0, 1]
    #             if not np.isnan(r):
    #                 corrs.append(r)
    #         quality[i] = np.mean(corrs) if corrs else 0.0

    #     keep_mask = quality >= CYCLE_CORR_THRESHOLD
    #     n_dropped = int((~keep_mask).sum())
    #     if n_dropped > 0:
    #         dropped_indices = np.where(~keep_mask)[0]
    #         print(f"  Artifact rejection: dropping {n_dropped} cycle(s) "
    #               f"(indices {list(dropped_indices)}, "
    #               f"corr = {quality[~keep_mask].round(2)})")
    #         for m in MUSCLES:
    #             cycles_arr[m] = cycles_arr[m][keep_mask]
    #         n_cycles = int(keep_mask.sum())
    #     print(f"  Cycles after artifact rejection: {n_cycles}")

    # ── 2d. Mean ± SD (Gaussian-smoothed) ────────────────────────────────────
    stats = {}
    for m in MUSCLES:
        arr    = cycles_arr[m]
        mean_s = gaussian_filter1d(arr.mean(axis=0),        sigma=1)
        sd_s   = gaussian_filter1d(arr.std(axis=0, ddof=1 if len(arr) > 1 else 0), sigma=1)
        stats[m] = {"mean": mean_s, "sd": sd_s, "n": len(arr)}

    # ── 2e. APDM – P90 per muscle ────────────────────────────────────────────
    apdm_p90 = {}
    for m in MUSCLES:
        apdm_p90[m] = compute_apdm_p90(cycles_arr[m], percentile=APDM_PERCENTILE)

    # Stash for cross-condition plot
    all_stats[cond_label] = {"stats": stats, "n": n_cycles}

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 6 : Full timeseries with cycle-boundary overlay
    # ──────────────────────────────────────────────────────────────────────────
    fig6, axes6 = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.18, wspace=0.05)
    fig6.suptitle(f"Subject01 — {cond_label} — EMG envelopes with s2s-cycle "
        f"segmentation  (EMG based, n = {n_cycles})",
        fontsize=23, fontweight="bold", y=0.99,
    )

    for ax_idx, muscle in enumerate(PLOT_ORDER):
        ax = axes6[ax_idx // 4][ax_idx % 4]

        ax.plot(time_s, s2s_norm[muscle] * 100,
                color=SEG_ENV_COLOR, linewidth=0.55, alpha=0.95,
                label="EMG envelope" if ax_idx == 0 else None)

        first_start, first_end = True, True
        for _, row in cond_seg.iterrows():
            ax.axvline(row["Start_Time_Sec"], color=SEG_START_COLOR,
                       linestyle="--", linewidth=0.7, alpha=0.6,
                       label="Cycle start" if (ax_idx == 0 and first_start) else None)
            ax.axvline(row["End_Time_Sec"], color=SEG_END_COLOR,
                       linestyle="--", linewidth=0.7, alpha=0.6,
                       label="Cycle end" if (ax_idx == 0 and first_end) else None)
            first_start = False
            first_end   = False

        ax.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
        ax.set_ylim(-2, 100)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if ax_idx % 4 == 0:
            ax.set_ylabel("%MVC", fontsize=12, fontweight="bold")
        if ax_idx // 4 == 3:
            ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")

    # axes6[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(top=0.94)       # pull plots up closer to title
    out6 = os.path.join(OUT_DIR, f"Fig6_EMG_FullTimeseries_Segmented_{file_tag}.png")
    plt.savefig(out6, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig6)
    print(f"  ✓ Fig6 (segmented timeseries) → {out6}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 5 : Individual cycles overlaid with mean
    # ──────────────────────────────────────────────────────────────────────────
    indiv_alpha = max(0.18, min(0.45, 4.0 / max(n_cycles, 1)))

    fig5, axes5 = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.18, wspace=0.05)
    fig5.suptitle(
        f"Subject01 — {cond_label} — Individual s2s cycles with mean overlay",
        fontsize=23, fontweight="bold", y=0.99,
    )

    for ax_idx, muscle in enumerate(PLOT_ORDER):
        ax  = axes5[ax_idx // 4][ax_idx % 4]
        arr = cycles_arr[muscle]
        st  = stats[muscle]

        for cyc in arr:
            ax.plot(pct_axis, cyc, color=INDIV_COLOR, linewidth=0.7,
                    alpha=indiv_alpha, zorder=2)
        ax.plot(pct_axis, st["mean"], color=MEAN_COLOR, linewidth=2.0,
                zorder=3, label="Mean")

        # ax.axvline(50, color="gray", linestyle="--", linewidth=1.0,
        #            alpha=0.8, zorder=4)

        # P90 annotation
        # p90_val = apdm_p90[muscle]
        # ax.axhline(p90_val, color="#8B0000", linestyle=":", linewidth=0.8,
        #            alpha=0.7, label=f"P90={p90_val:.2f}" if ax_idx == 0 else None)

        ax.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if ax_idx % 4 == 0:
            ax.set_ylabel("Norm EMG", fontsize=12, fontweight="bold")
        if ax_idx // 4 == 3:
            ax.set_xlabel("S2S cycle (%)", fontsize=12, fontweight="bold")

    # axes5[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(top=0.94)       # pull plots up closer to title
    out5 = os.path.join(OUT_DIR, f"Fig5_EMG_S2S_Cycle_Individual_{file_tag}.png")
    plt.savefig(out5, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig5)
    print(f"  ✓ Fig5 (individual cycles)   → {out5}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 4 : Cycle-averaged mean ± SD
    # ──────────────────────────────────────────────────────────────────────────
    fig4, axes4 = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
    plt.subplots_adjust(hspace=0.18, wspace=0.05)
    fig4.suptitle(f"Subject01 — {cond_label} — EMG (Global MVC) mean ± SD",
        fontsize=23, fontweight="bold", y=0.99,
    )

    for ax_idx, muscle in enumerate(PLOT_ORDER):
        ax    = axes4[ax_idx // 4][ax_idx % 4]
        mn    = stats[muscle]["mean"]
        sd    = stats[muscle]["sd"]
        upper = mn + sd
        lower = np.maximum(mn - sd, 0)

        ax.fill_between(pct_axis, lower, upper, color=CYCLE_FILL,
                        alpha=0.7, linewidth=0)
        ax.plot(pct_axis, mn, color=CYCLE_LINE, linewidth=1.8)
        # ax.axvline(50, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

        # P90 reference line
        # p90_val = apdm_p90[muscle]
        # ax.axhline(p90_val, color="#8B0000", linestyle=":", linewidth=0.9,
        #            alpha=0.8, label=f"P90={p90_val:.2f}" if ax_idx == 0 else None)

        ax.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if ax_idx % 4 == 0:
            ax.set_ylabel("Norm EMG", fontsize=12, fontweight="bold")
        if ax_idx // 4 == 3:
            ax.set_xlabel("S2S cycle (%)", fontsize=12, fontweight="bold")

    # axes4[0][0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(top=0.94)
    out4 = os.path.join(OUT_DIR, f"Fig4_EMG_S2S_Cycle_Mean_SD_{file_tag}.png")
    plt.savefig(out4, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig4)
    print(f"  ✓ Fig4 (mean ± SD)           → {out4}")

    # ──────────────────────────────────────────────────────────────────────────
    # Save cycle-average data CSV (mean, SD, P90)
    # ──────────────────────────────────────────────────────────────────────────
    rows = []
    for m in PLOT_ORDER:
        p90 = apdm_p90[m]
        for pct, mn_val, sd_val in zip(pct_axis,
                                       stats[m]["mean"],
                                       stats[m]["sd"]):
            rows.append({
                "Condition":    cond_label,
                "Muscle":       m,
                "Cycle_%":      round(pct, 1),
                "Mean_normEMG": round(mn_val, 5),
                "SD_normEMG":   round(sd_val, 5),
                f"P{APDM_PERCENTILE}_APDM": round(p90, 5),
            })
    csv_path = os.path.join(OUT_DIR, f"EMG_CycleAvg_Subject01_{file_tag}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✓ Cycle-average CSV          → {csv_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIG 8 : Cycle-averaged mean ± SD (0-48% and 52-100% regions)
    # ──────────────────────────────────────────────────────────────────────────
    fig8 = plt.figure(figsize=(20, 14))
    fig8.suptitle(f"Subject01 — {cond_label} — EMG mean ± SD",
                  fontsize=23, fontweight="bold", y=0.99)

    # Create a 4x4 GridSpec, each cell will be split into two sub-cells
    gs_main = fig8.add_gridspec(4, 4, hspace=0.18, wspace=0.15)

    for ax_idx, muscle in enumerate(PLOT_ORDER):
        # Create sub-gridspec for the two segments (0-48 and 52-100)
        subgs = gs_main[ax_idx // 4, ax_idx % 4].subgridspec(1, 2, wspace=0)
        
        ax_left = fig8.add_subplot(subgs[0, 0])
        ax_right = fig8.add_subplot(subgs[0, 1], sharey=ax_left)
        
        mn = stats[muscle]["mean"]
        sd = stats[muscle]["sd"] 
        upper = mn + sd
        lower = np.maximum(mn - sd, 0)

        # Plot 0-48% on the left plot
        ax_left.fill_between(pct_axis[:49], lower[:49], upper[:49], 
                             color=CYCLE_FILL, alpha=0.7, linewidth=0)
        ax_left.plot(pct_axis[:49], mn[:49], color=CYCLE_LINE, linewidth=1.8)
        
        # Plot 52-100% on the right plot
        ax_right.fill_between(pct_axis[52:], lower[52:], upper[52:], 
                              color=CYCLE_FILL, alpha=0.7, linewidth=0)
        ax_right.plot(pct_axis[52:], mn[52:], color=CYCLE_LINE, linewidth=1.8)
        
        # Formatting
        for ax in [ax_left, ax_right]:
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=10)
            
        # Sync ticks and remove inner labels
        ax_left.set_xticks([0, 15, 48])
        ax_right.set_xticks([52, 100])
        ax_right.tick_params(axis='y', left=False, labelleft=False) # Hide shared Y-axis
        
        # Titles and Labels
        ax_left.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
        ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        
        if ax_idx % 4 == 0:
            ax_left.set_ylabel("Norm EMG", fontsize=12, fontweight="bold")
        # if ax_idx // 4 == 3:
        #     # Center the label across the two sub-plots
        #     ax_left.text(15, -0.15, "S2S cycle (%)", transform=ax_left.transAxes, 
        #                  ha='center', fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.04, 1, 1]) 
    plt.subplots_adjust(top=0.94)  
    out8 = os.path.join(OUT_DIR, f"Fig8_EMG_S2S_Cycle_Mean_SD_Zoom_{file_tag}.png")
    plt.savefig(out8, dpi=180, bbox_inches="tight")
    # #plt.show()
    plt.close(fig8)
    print(f"  ✓ Fig8 (zoomed mean ± SD)    → {out8}")




# ── FIG 7: All conditions mean ± SD on one 4×4 grid ──────────────────────────
CONDITION_COLORS = {
    "No Exo":    "#0b8ae6",   # orange
    "No Torque": "#f70000",   # green
    "NN Torque":   "#00ff00",   # red
}
 
fig7, axes7 = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
plt.subplots_adjust(hspace=0.18, wspace=0.05)
fig7.suptitle(("Subject01 — EMG (Global MVC) mean by condition"),
    fontsize=23, fontweight="bold", y=0.99, 
)
 
for ax_idx, muscle in enumerate(PLOT_ORDER):
    ax = axes7[ax_idx // 4][ax_idx % 4]
 
    for cond_label, _, _, _, _ in TRIALS:
        if cond_label not in all_stats:
            continue
        st    = all_stats[cond_label]["stats"][muscle]
        n     = all_stats[cond_label]["n"]
        color = CONDITION_COLORS.get(cond_label, "gray")
        mn    = st["mean"]
        sd    = st["sd"]
        upper = mn + sd
        lower = np.maximum(mn - sd, 0)
 
        ax.fill_between(pct_axis, lower, upper,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(pct_axis, mn, color=color, linewidth=1.8,
                label=f"{cond_label}")
 
    # ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
    ax.set_xlim(0, 100)
    # ax.set_ylim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if ax_idx % 4 == 0:
        ax.set_ylabel("Norm EMG", fontsize=12, fontweight="bold")
    if ax_idx // 4 == 3:
        ax.set_xlabel("S2S cycle (%)", fontsize=12, fontweight="bold")


# Single horizontal legend at the bottom of the figure
handles, labels = axes7[0][0].get_legend_handles_labels()
fig7.legend(handles, labels, loc="lower center", ncol=len(TRIALS),
            fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.01))
 
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.subplots_adjust(top=0.94)       # pull plots up closer to title
out7 = os.path.join(OUT_DIR, "Fig7_EMG_MeanSD_AllConditions.png")
plt.savefig(out7, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig7)
print(f"  ✓ Fig7 (cross-condition mean ± SD) → {out7}")



# ──────────────────────────────────────────────────────────────────────────
# FIG 9 : All conditions mean ± SD (0-48% and 52-100% regions)
# ──────────────────────────────────────────────────────────────────────────

fig9 = plt.figure(figsize=(20, 14))
fig9.suptitle(f"Subject01 — EMG mean by condition",
              fontsize=23, fontweight="bold", y=0.99)
gs_main = fig9.add_gridspec(4, 4, hspace=0.18, wspace=0.15)

for ax_idx, muscle in enumerate(PLOT_ORDER):
    # Create sub-gridspec for the two segments
    subgs = gs_main[ax_idx // 4, ax_idx % 4].subgridspec(1, 2, wspace=0)
    ax_left = fig9.add_subplot(subgs[0, 0])
    ax_right = fig9.add_subplot(subgs[0, 1], sharey=ax_left)
    
    # Iterate through all conditions to plot them on the same axes
    for cond_label, _, _, _, _ in TRIALS:
        if cond_label not in all_stats: continue
        
        st    = all_stats[cond_label]["stats"][muscle]
        color = CONDITION_COLORS.get(cond_label, "gray")
        mn    = st["mean"]
        sd    = st["sd"]
        
        # Plot segments for each condition
        for ax, rng in [(ax_left, slice(0, 49)), (ax_right, slice(52, 101))]:
            ax.fill_between(pct_axis[rng], (mn-sd)[rng], (mn+sd)[rng], 
                            color=color, alpha=0.1, linewidth=0)
            ax.plot(pct_axis[rng], mn[rng], color=color, linewidth=1.5, label=cond_label)

    # Formatting for this subplot
    for ax in [ax_left, ax_right]:
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=10)

    if ax_idx // 4 == 3:  # Check if the subplot is in the bottom row
        ax_left.set_xticks([0, 15, 48])
        ax_right.set_xticks([52, 100])
    else:
        ax_left.set_xticks([])
        ax_right.set_xticks([])

    ax_right.tick_params(axis='y', left=False, labelleft=False)
    ax_left.set_title(muscle, fontsize=12, fontweight="bold", loc="right")

    # ax.set_title(muscle, fontsize=12, fontweight="bold", pad=3)
    # ax.set_ylim(0, 1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # if ax_idx % 4 == 0:
    #     ax.set_ylabel("Norm EMG", fontsize=12, fontweight="bold")
    if ax_idx // 4 == 3:
        ax.set_xlabel("S2S cycle (%)", fontsize=12, fontweight="bold", loc="left")

# Add legend and save
# Single horizontal legend at the bottom of the figure
handles, labels = ax_left.get_legend_handles_labels()
fig9.legend(handles, labels, loc="lower center", ncol=len(TRIALS),
            fontsize=20, frameon=False, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.subplots_adjust(top=0.94)       # pull plots up closer to title
out9 = os.path.join(OUT_DIR, "Fig9_EMG_S2S_Cycle_Mean_SD_Zoom_AllConditions.png")
plt.savefig(out9, dpi=180, bbox_inches="tight")
# #plt.show()
plt.close(fig9)
print(f"  ✓ Fig9 (zoomed mean ± SD all conditions) → {out9}")