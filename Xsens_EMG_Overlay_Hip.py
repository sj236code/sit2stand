"""Visual shift tester: overlay Xsens HIP FLEXION/EXTENSION ANGLE and EMG hip-muscle activity.

Place this file beside the Subject01 S2S CSV and Xsens XLSX files.
Run: python Test_Xsens_EMG_Shifts.py
Outputs: ./test_shifts/

EDIT OFFSETS below and rerun. Positive offset moves Xsens to the RIGHT in EMG time.
This is a VISUAL diagnostic only; it does not select a shift or segment cycles.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save images without showing windows
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'test_shifts_2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CHANGE THESE NUMBERS AND RERUN. Positive = Xsens moves later on EMG timeline.
OFFSETS_SECONDS = {
    'NoExo': 0.10,
    'NoTorque': 9.20,
    'NN': 2.05,
}

FILES = {
    'NoExo': ('EMG_Recording_Subject01_NoExo_s2s.csv', 'Subject01_NoExo_s2s.xlsx'),
    'NoTorque': ('EMG_Recording_Subject01_NoTorque_s2s.csv', 'Subject01_NoTorque_s2s.xlsx'),
    'NN': ('EMG_Recording_Subject01_NN_s2s_10nm.csv', 'Subject01_NN_s2s_10nm.xlsx'),
}

# These are hip-related EMG channels, not a measured hip angle.
EMG_GROUPS = {
    'Hip extensors (GMax/BF/ST)': ['Env_GMax_R', 'Env_GMax_L', 'Env_BF_R', 'Env_BF_L', 'Env_ST_R', 'Env_ST_L'],
    'Hip flexors (IL/TFL)': ['Env_IL_R', 'Env_IL_L', 'Env_TFL_R', 'Env_TFL_L'],
}
EMG_CHANNELS = list(dict.fromkeys(c for group in EMG_GROUPS.values() for c in group))
XSENS_HZ = 100.0  # Xsens files recorded at 100 Hz; check General Information if changed
EMG_SMOOTH_SECONDS = 0.12
HIP_SMOOTH_SECONDS = 0.08
# Each plot overlays hip angle (degrees, LEFT axis) with EMG activity (scaled 0-1, RIGHT axis).
# Zoom plots make the relative timing of sit/stand events easier to inspect.
ZOOM_WINDOWS = [(0, 45), (45, 90), (90, 135), (135, 190)]


def find_file(filename):
    p = BASE_DIR / filename
    if p.exists():
        return p
    # Some exports have a trailing (1) copy suffix.
    options = list(BASE_DIR.glob(p.stem + ' (*' + p.suffix))
    if len(options) == 1:
        print(f'  Using alternate file: {options[0].name}')
        return options[0]
    raise FileNotFoundError(f'Cannot find {filename} in {BASE_DIR}. Check FILES at top of script.')


def load_emg(path):
    df = pd.read_csv(path, usecols=['timestamp'] + EMG_CHANNELS)
    # The timestamp column is milliseconds; duplicates are normal due to export precision.
    t = (df['timestamp'].to_numpy(dtype=float) - float(df['timestamp'].iloc[0])) / 1000.0
    if not np.isfinite(t).all() or t[-1] <= t[0]:
        raise ValueError(f'Invalid EMG timestamps in {path.name}')
    fs = (len(t) - 1) / (t[-1] - t[0])  # Avoid median(diff): duplicate timestamps yield zero.
    env = df[EMG_CHANNELS].to_numpy(dtype=float)
    activity = {}
    for label, channels in EMG_GROUPS.items():
        indices = [EMG_CHANNELS.index(c) for c in channels]
        signal = np.nanmean(env[:, indices], axis=1)
        activity[label] = gaussian_filter1d(signal, sigma=max(1, round(EMG_SMOOTH_SECONDS * fs)))
    # Downsample for efficient plotting without changing the original time reference.
    step = max(1, round(fs / 100))
    t = t[::step]
    for label, signal in activity.items():
        signal = signal[::step]
        lo = np.nanpercentile(signal, 5)
        hi = np.nanpercentile(signal, 99)
        activity[label] = np.clip((signal - lo) / max(hi - lo, 1e-12), 0, 1.2)
    return t, activity


def load_hip(path):
    df = pd.read_excel(path, sheet_name='Joint Angles ZXY',
                       usecols=['Frame', 'Right Hip Flexion/Extension', 'Left Hip Flexion/Extension'],
                       engine='openpyxl')
    t = (df['Frame'].to_numpy(dtype=float) - float(df['Frame'].iloc[0])) / XSENS_HZ
    right = df['Right Hip Flexion/Extension'].to_numpy(dtype=float)
    left = df['Left Hip Flexion/Extension'].to_numpy(dtype=float)
    hip = gaussian_filter1d((right + left) / 2, sigma=HIP_SMOOTH_SECONDS * XSENS_HZ)
    return t, hip


def plot_overlay(condition, t_emg, activity, t_xsens, hip, offset, bounds, filename):
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax2 = ax.twinx()
    ax.plot(t_xsens + offset, hip, color='#176bb0', lw=2, label='Xsens mean hip flexion/extension (deg)')
    for label, color in [('Hip extensors (GMax/BF/ST)', '#d45520'), ('Hip flexors (IL/TFL)', '#25935b')]:
        ax2.plot(t_emg, activity[label], color=color, lw=1.5, alpha=0.85, label='EMG ' + label + ' (scaled)')
    ax.set_xlabel('EMG-relative time (seconds)')
    ax.set_ylabel('Xsens hip flexion/extension (degrees)', color='#176bb0')
    ax2.set_ylabel('EMG hip muscle activity (scaled)')
    ax.set_ylim(np.nanmin(hip) - 10, np.nanmax(hip) + 10)
    ax2.set_ylim(-0.05, 1.25)
    ax.set_xlim(*bounds)
    ax.grid(alpha=0.22)
    ax.set_title(f'{condition} | Xsens hip ANGLE + EMG hip muscles | Xsens shift {offset:+.2f} s')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'  Saved: {out.name}')


def main():
    print(f'Writing diagnostic plots to: {OUT_DIR}')
    rows = []
    for condition, (emg_file, xsens_file) in FILES.items():
        print(f'\n--- {condition} ---')
        emg_path = find_file(emg_file)
        xsens_path = find_file(xsens_file)
        t_emg, activity = load_emg(emg_path)
        t_xsens, hip = load_hip(xsens_path)
        offset = float(OFFSETS_SECONDS[condition])
        plot_overlay(condition, t_emg, activity, t_xsens, hip, offset,
                     (0, max(t_emg[-1], t_xsens[-1] + offset)),
                     f'{condition}_FULL_shift_{offset:+.2f}s.png')
        for start, end in ZOOM_WINDOWS:
            if start >= max(t_emg[-1], t_xsens[-1] + offset):
                continue
            plot_overlay(condition, t_emg, activity, t_xsens, hip, offset,
                         (start, min(end, max(t_emg[-1], t_xsens[-1] + offset))),
                         f'{condition}_ZOOM_{start:03d}-{end:03d}s.png')
        # Export both signals on a common 100-Hz EMG-relative timeline for manual inspection.
        grid = np.arange(0, t_emg[-1], 0.01)
        common = pd.DataFrame({
            'EMG_time_s': grid,
            'EMG_hip_extensors_scaled': np.interp(grid, t_emg, activity['Hip extensors (GMax/BF/ST)'], left=np.nan, right=np.nan),
            'EMG_hip_flexors_scaled': np.interp(grid, t_emg, activity['Hip flexors (IL/TFL)'], left=np.nan, right=np.nan),
            'Xsens_hip_angle_deg': np.interp(grid, t_xsens + offset, hip, left=np.nan, right=np.nan),
        })
        common.to_csv(OUT_DIR / f'{condition}_overlay_data.csv', index=False)
        rows.append({'condition': condition, 'xsens_shift_seconds': offset,
                     'emg_duration_seconds': t_emg[-1], 'xsens_duration_seconds': t_xsens[-1],
                     'emg_file': emg_path.name, 'xsens_file': xsens_path.name})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'shifts_used.csv', index=False)
    print('\nDONE. Change OFFSETS_SECONDS at the top and rerun to compare different alignments.')
    print('NOTE: These offsets are provisional. Peaks in EMG and hip angle need not coincide.')


if __name__ == '__main__':
    main()
