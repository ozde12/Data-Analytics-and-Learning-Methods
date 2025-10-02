import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict
from sklearn.metrics import silhouette_score


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def preprocess_running(trimmed_root, all_data, window_size=3.0, show_plots=True):
    activity = "running"
    for folder_name, folder_data in all_data[activity].items():
        out_dir = os.path.join(trimmed_root, activity, folder_name)

        # Remove old processed folder if it exists
        if os.path.exists(out_dir):
            print(f"Reprocessing {activity} | Folder: {folder_name} (removing old results)")
            shutil.rmtree(out_dir)
        else:
            print(f"\n Processing {activity} | Folder: {folder_name}")

        # 1. Single-file (Accelerometer only)
        df = folder_data["Accelerometer.csv"]

        time = df["seconds_elapsed"].values
        y_values = df["y"].values
        sample_rate = len(time) / (time[-1] - time[0])
        step = int(window_size * sample_rate)
        stride = int(step * 0.5)  # 50% overlap

        trim_start, trim_end = detect_trim_bounds_rms_intersection(time,
                                                                   y_values,
                                                                   step,
                                                                   stride,
                                                                   frac_start=0.85,   # start when RMS passes 20% of peak
                                                                   frac_end=0.7,     # end when RMS falls below 20% of peak
                                                                   smooth=5,
                                                                   debug=show_plots)  # debug=False to disable plots
        # Now actually process the folder
        n_win, out_dir = process_folder(folder_data, 
                                        activity, 
                                        folder_name, 
                                        trimmed_root,
                                        trim_start=trim_start, 
                                        trim_end=trim_end,
                                        window_size=window_size,
                                        overlap=0.5)

        print(f"  Trim {trim_start:.2f} → {trim_end:.2f}, {n_win} windows, saved in {out_dir}")


def preprocess_walking(trimmed_root, all_data, window_size=3.0, show_plots=True):
    activity = "walking"
    for folder_name, folder_data in all_data[activity].items():
        out_dir = os.path.join(trimmed_root, activity, folder_name)

        # Remove old processed folder if it exists
        if os.path.exists(out_dir):
            print(f"Reprocessing {activity} | Folder: {folder_name} (removing old results)")
            shutil.rmtree(out_dir)

        else:
            print(f"\n Processing {activity} | Folder: {folder_name} ")

        #  Single-file (Accelerometer only)
        df = folder_data["Accelerometer.csv"]

        time = df["seconds_elapsed"].values
        y_values = df["y"].values
        sample_rate = len(time) / (time[-1] - time[0])
        step = int(window_size * sample_rate)
        stride = int(step * 0.5)  # 50% overlap

        trim_start, trim_end = detect_trim_bounds_rms_intersection(time,
                                                                   y_values,
                                                                   step,
                                                                   stride,
                                                                   frac_start=0.85,   # start when RMS passes 20% of peak
                                                                   frac_end=0.7,      # end when RMS falls below 20% of peak
                                                                   smooth=5,
                                                                   debug=show_plots)        # debug=False to disable plots
        # Now actually process the folder
        n_win, out_dir = process_folder(folder_data,
                                        activity, 
                                        folder_name, 
                                        trimmed_root,
                                        trim_start=trim_start,
                                        trim_end=trim_end,
                                        window_size=window_size,
                                        overlap=0.5)
        
        print(f"  Trim {trim_start:.2f} → {trim_end:.2f}, {n_win} windows, saved in {out_dir}")


def preprocess_stairs(trimmed_root, all_data, window_size=3.0, show_plots=True):

    activity = "stairs"
    out_dir = os.path.join(trimmed_root, activity)

    # Wipe the entire stairs output, then recreate it
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for folder_name, folder_data in all_data[activity].items():
        print(f"\n Processing {activity} | Folder: {folder_name} ")
        # Normalize time across all CSVs in this folder 
        for name, df in folder_data.items():
            if "seconds_elapsed" in df.columns:
                df["seconds_elapsed"] = df["seconds_elapsed"] - df["seconds_elapsed"].iloc[0]

        # Detect trim from Gravity.csv 
        df_gravity = folder_data["Gravity.csv"]

        time = df_gravity["seconds_elapsed"].values
        y_values = df_gravity["y"].values
        sample_rate = len(time) / (time[-1] - time[0])
        step = int(window_size * sample_rate)
        stride = int(step * 0.5)  # 50% overlap

        trim_start, trim_end = detect_trim_bounds_rms_intersection(time,
                                                                y_values,
                                                                step,
                                                                stride,
                                                                frac_start=0.98,
                                                                frac_end=0.97,
                                                                smooth=2,
                                                                debug=False)

        print(f"{folder_name}: Trim {trim_start:.2f} → {trim_end:.2f}")

        # Plot accelerometer with trim markers
        df_acc = folder_data["Accelerometer.csv"]
        plt.figure(figsize=(12, 6))
        plt.plot(df_acc["seconds_elapsed"], df_acc["x"], label="x", alpha=0.7)
        plt.plot(df_acc["seconds_elapsed"], df_acc["y"], label="y", alpha=0.7)
        plt.plot(df_acc["seconds_elapsed"], df_acc["z"], label="z", alpha=0.7)

        plt.axvline(trim_start, color="g", linewidth=2, label=f"Trim start {trim_start:.2f}s")
        plt.axvline(trim_end, color="r", linewidth=2, label=f"Trim end {trim_end:.2f}s")

        plt.title(f"Accelerometer Trim - {activity} | {folder_name}")
        plt.xlabel("Time [s]")
        plt.ylabel("Accelerometer values")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(out_dir, f"{folder_name}_accelerometer_trim.png")
        plt.savefig(plot_path, dpi=150)
        if show_plots:
            plt.show()
        plt.close()
        #print(f"Saved accelerometer plot to {plot_path}")

        # Slice all raw CSVs into windows (per window folder)
        step = int(window_size * sample_rate)
        stride = int(step * 0.5)

        # Use accelerometer to decide window ranges
        df_acc_trimmed = df_acc[(df_acc["seconds_elapsed"] >= trim_start) &
                                (df_acc["seconds_elapsed"] <= trim_end)]
        acc_time = df_acc_trimmed["seconds_elapsed"].values
        start_idx = np.searchsorted(acc_time, trim_start)
        end_idx = np.searchsorted(acc_time, trim_end)

        window_ranges = []
        for i, start in enumerate(range(start_idx, end_idx - step + 1, stride)):
            end = start + step

            # Define a window directory
            win_dir = os.path.join(out_dir, f"{folder_name}_window_{i+1}")
            os.makedirs(win_dir, exist_ok=True)

            # Save trimmed data for all files into this window
            for file_name, df in folder_data.items():
                if "seconds_elapsed" not in df.columns:
                    continue

                df_trimmed = df[(df["seconds_elapsed"] >= trim_start) &
                                (df["seconds_elapsed"] <= trim_end)]

                window_df = df_trimmed.iloc[start:end]
                if len(window_df) < step:
                    continue

                file_out = os.path.join(win_dir, file_name)
                window_df.to_csv(file_out, index=False)

            # Record ranges for summary plot
            win_start = df_acc_trimmed.iloc[start]["seconds_elapsed"]
            win_end = df_acc_trimmed.iloc[end-1]["seconds_elapsed"]
            window_ranges.append((win_start, win_end))

        # Summary plot with shaded windows 
        plt.figure(figsize=(12, 6))
        plt.plot(df_acc["seconds_elapsed"], df_acc["x"], label="x", alpha=0.7)
        plt.plot(df_acc["seconds_elapsed"], df_acc["y"], label="y", alpha=0.7)
        plt.plot(df_acc["seconds_elapsed"], df_acc["z"], label="z", alpha=0.7)

        plt.axvline(trim_start, color="red", linestyle="--", label=f"Trim start {trim_start:.2f}s")
        plt.axvline(trim_end, color="blue", linestyle="--", label=f"Trim end {trim_end:.2f}s")

        for win_start, win_end in window_ranges:
            plt.axvspan(win_start, win_end, color="gray", alpha=0.15)

        plt.title(f"{activity} - {folder_name} - Accelerometer (Trim + {window_size:.0f}s windows, 50% overlap)")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [m/s²]")
        plt.legend()
        plt.tight_layout()

        summary_path = os.path.join(out_dir, f"{folder_name}_summary_plot.png")
        plt.savefig(summary_path, dpi=150)
        if show_plots:
            plt.show()
        plt.close()
        #print(f"Saved summary plot to {summary_path}")

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def trim_and_window(df, activity, folder_name, out_root, window_size=5, overlap=0.5):
    """
    Detect trim start & end from Accelerometer.csv, slice into overlapping windows
    only between those bounds, save windows as CSV, and save a summary plot.
    """
    time = df["seconds_elapsed"].values
    y_values = df["y"].values

    # Parameters
    sample_rate = len(time) / (time[-1] - time[0])
    step = int(window_size * sample_rate)
    stride = int(step * (1 - overlap))

    trim_start, trim_end = detect_trim_bounds_rms_intersection(
        time, y_values, step, stride,
        baseline_windows=10,
        factor_start=1.3,          # tweak
        end_method="peak",
        peak_end_frac=0.65,        # tweak (0.6–0.75 works well)
        required_hits_start=3,
        required_hits_end=3,
        debug=False
    )

    start_idx = np.searchsorted(time, trim_start)
    end_idx   = np.searchsorted(time, trim_end)
    
    for i, start in enumerate(range(start_idx, end_idx - step + 1, stride)):
        end = start + step
        window_df = df.iloc[start:end]
        overlapping_windows.append(window_df)

        # Save each window as CSV
        out_path = os.path.join(out_dir, f"window_{i+1}.csv")
        window_df.to_csv(out_path, index=False)


    # Output directory
    out_dir = os.path.join(out_root, activity, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # Slice into overlapping windows **only within trim_start–trim_end**
    overlapping_windows = []
    start_idx = np.searchsorted(time, trim_start)
    end_idx = np.searchsorted(time, trim_end)
    for i, start in enumerate(range(start_idx, end_idx - step + 1, stride)):
        end = start + step
        window_df = df.iloc[start:end]
        overlapping_windows.append(window_df)

        # Save each window as CSV
        out_path = os.path.join(out_dir, f"window_{i+1}.csv")
        window_df.to_csv(out_path, index=False)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, df["x"], label="x")
    plt.plot(time, df["y"], label="y")
    plt.plot(time, df["z"], label="z")

    # Mark trim start & end
    plt.axvline(trim_start, color="red", linestyle="--", label="Trim start")
    plt.axvline(trim_end, color="blue", linestyle="--", label="Trim end")

    # Shade overlapping windows (only between trim_start and trim_end)
    for window_df in overlapping_windows:
        win_start = window_df["seconds_elapsed"].iloc[0]
        win_end = window_df["seconds_elapsed"].iloc[-1]
        plt.axvspan(win_start, win_end, color="gray", alpha=0.1)

    plt.title(f"{activity} - {folder_name} - Accelerometer.csv (Trim + Overlap)")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "summary_plot.png"))
    plt.close()

    return trim_start, trim_end, len(overlapping_windows), out_dir

def process_folder(folder_data, activity, folder_name, out_root,
                   trim_start=None, trim_end=None,
                   window_size=3.0, overlap=0.5):
    """
    Create time windows between trim_start and trim_end.
    If not provided, detect them from Accelerometer.csv.
    """
    df_acc = folder_data["Accelerometer.csv"]
    time = df_acc["seconds_elapsed"].values
    y_values = df_acc["y"].values

    sample_rate = len(time) / (time[-1] - time[0])
    step = int(window_size * sample_rate)
    stride = int(step * (1 - overlap))

    # --- Trim detection if not provided ---
    if trim_start is None or trim_end is None:
        trim_start, trim_end = detect_trim_bounds_rms_intersection(
            time, y_values, step, stride,
            frac_start=0.8, frac_end=0.7,
            smooth=5, debug=False
        )

    # --- Build windows ---
    windows = []
    start_idx = np.searchsorted(time, trim_start)
    end_idx   = np.searchsorted(time, trim_end)
    for i, start in enumerate(range(start_idx, end_idx - step + 1, stride)):
        end = start + step
        windows.append((time[start], time[end - 1]))

    # --- Save windows ---
    out_dir = os.path.join(out_root, activity)
    os.makedirs(out_dir, exist_ok=True)

    for i, (win_start, win_end) in enumerate(windows):
        win_dir = os.path.join(out_dir, f"{folder_name}_window_{i+1}")
        os.makedirs(win_dir, exist_ok=True)
        for file_name, file_df in folder_data.items():
            if "seconds_elapsed" not in file_df.columns:
                continue
            mask = (file_df["seconds_elapsed"] >= win_start) & (file_df["seconds_elapsed"] <= win_end)
            file_df[mask].to_csv(os.path.join(win_dir, file_name), index=False)

    return len(windows), out_dir

def detect_trim_bounds_rms_intersection(time,
                                        y_values,
                                        step,
                                        stride,
                                        frac_start: float = 0.2,   # fraction of peak for start
                                        frac_end: float = 0.2,     # fraction of peak for end
                                        smooth: int = 5,
                                        debug: bool = False):
    """
    Detect trim start & end based on RMS intersections with the defined tresholds.
    """
    
    # RMS trace by sliding window  
    rms_vals, rms_times = [], []
    for s in range(0, len(y_values) - step + 1, stride):
        e = s + step
        w = y_values[s:e]
        rms_vals.append(np.sqrt(np.mean(w**2)))
        rms_times.append(time[s])
    rms_vals = np.asarray(rms_vals)
    rms_times = np.asarray(rms_times)

    # Smooth RMS by moving average  
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        rms_vals_smooth = np.convolve(rms_vals, kernel, mode="same")
    else:
        rms_vals_smooth = rms_vals

    #  Peak-based thresholds 
    peak = np.max(rms_vals_smooth)
    thr_start = frac_start * peak
    thr_end   = frac_end * peak

    #  Find trim_start = first intersection going up 
    trim_start = rms_times[0]
    for i in range(1, len(rms_vals_smooth)):
        if rms_vals_smooth[i-1] < thr_start and rms_vals_smooth[i] >= thr_start:
            trim_start = rms_times[i]
            break

    #  Find trim_end = last intersection going down 
    trim_end = rms_times[-1]
    for i in range(len(rms_vals_smooth)-1, 0, -1):
        if rms_vals_smooth[i-1] > thr_end and rms_vals_smooth[i] <= thr_end:
            trim_end = rms_times[i]
            break

    # Debug 
    if debug:
        plt.figure(figsize=(12, 4))
        plt.plot(rms_times, rms_vals, label="RMS Energy", alpha=0.6)
        plt.plot(rms_times, rms_vals_smooth, label="Smoothed RMS", linewidth=2)
        plt.axhline(thr_start, color="red", linestyle="--", label=f"Start thr ({thr_start:.2f})")
        plt.axhline(thr_end, color="purple", linestyle="--", label=f"End thr ({thr_end:.2f})")
        plt.axvline(trim_start, color="green", linestyle="--", label=f"Trim start {trim_start:.1f}s")
        plt.axvline(trim_end, color="blue", linestyle="--", label=f"Trim end {trim_end:.1f}s")
        plt.xlabel("Time [s]")
        plt.ylabel("RMS")
        plt.legend()
        plt.title("RMS intersection-based trim detection")
        plt.show()

    return trim_start, trim_end

def detect_trim_bounds_stairs(time, df_gravity, window_size=5, overlap=0.5,
                              start_frac=0.3, end_frac=0.2, smooth=5, debug=False,
                              out_dir=None, folder_name=None):
    """
    Detect and plot trim bounds for stairs activity using RMS of gravity signal.

    Parameters
    ----------
    time : array-like
        Time values (seconds_elapsed from Gravity.csv).
    df_gravity : DataFrame
        DataFrame with 'x','y','z' columns from Gravity.csv.
    window_size : float
        Window size in seconds for RMS calculation.
    overlap : float
        Fractional overlap between windows (0.5 = 50%).
    start_frac : float
        Fraction of peak RMS to mark activity start.
    end_frac : float
        Fraction of peak RMS to mark activity end.
    smooth : int
        Window size for moving average smoothing of RMS.
    debug : bool
        If True, generate summary plot.
    out_dir : str
        Path to save plot (inside trimmed_data/stairs/...).
    folder_name : str
        Current folder name for labeling/saving.

    Returns
    -------
    trim_start, trim_end : float
        Time values marking activity start and end.
    """

    # --- Compute overall magnitude of gravity ---
    g_mag = np.sqrt(df_gravity["x"]**2 + df_gravity["y"]**2 + df_gravity["z"]**2)

    # --- Sampling rate ---
    sample_rate = len(time) / (time[-1] - time[0])
    step = int(window_size * sample_rate)
    stride = int(step * (1 - overlap))

    # --- Compute RMS in windows ---
    rms_vals, rms_times = [], []
    for start in range(0, len(g_mag) - step, stride):
        segment = g_mag[start:start+step]
        rms_vals.append(np.sqrt(np.mean(segment**2)))
        rms_times.append(time[start + step//2])

    rms_vals = np.array(rms_vals)
    rms_times = np.array(rms_times)

    # --- Smooth RMS if needed ---
    if smooth > 1 and len(rms_vals) >= smooth:
        kernel = np.ones(smooth) / smooth
        rms_vals = np.convolve(rms_vals, kernel, mode="same")

    # --- Too few windows? fall back ---
    if len(rms_vals) < 3:
        return time[0], time[-1]

    # --- Thresholding ---
    peak = np.max(rms_vals)
    start_thresh = start_frac * peak
    end_thresh = end_frac * peak

    try:
        start_idx = np.where(rms_vals >= start_thresh)[0][0]
    except IndexError:
        start_idx = 0

    try:
        end_idx = np.where(rms_vals >= end_thresh)[0][-1]
        if end_idx >= len(rms_times):
            end_idx = len(rms_times) - 1
    except IndexError:
        end_idx = len(rms_times) - 1

    trim_start = rms_times[start_idx]
    trim_end = rms_times[end_idx]

    #  Debug / Summary plot 
    if debug:
        plt.figure(figsize=(12,6))
        # Plot raw gravity magnitude
        plt.plot(time, g_mag, alpha=0.4, label="Gravity magnitude")

        # Plot RMS curve
        plt.plot(rms_times, rms_vals, label="RMS (smoothed)", linewidth=2)

        # Thresholds
        plt.axhline(start_thresh, color="g", linestyle="--", label=f"Start threshold ({start_frac*100:.0f}% peak)")
        plt.axhline(end_thresh, color="r", linestyle="--", label=f"End threshold ({end_frac*100:.0f}% peak)")

        # Trim markers
        plt.axvline(trim_start, color="g", linewidth=2, label=f"Trim start {trim_start:.2f}s")
        plt.axvline(trim_end, color="r", linewidth=2, label=f"Trim end {trim_end:.2f}s")

        plt.title(f"Stairs Trim Detection - {folder_name}")
        plt.xlabel("Time [s]")
        plt.ylabel("Gravity / RMS")
        plt.legend()
        plt.tight_layout()

        # Save summary figure if out_dir provided
        if out_dir is not None and folder_name is not None:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{folder_name}_stairs_trim_summary.png")
            plt.savefig(save_path, dpi=150)
            print(f"Summary plot saved to {save_path}")

        plt.show()

    return trim_start, trim_end

# --------------------------------------------------------------------

def find_activity_center(acc_df, env_win_s=0.5, thresh_ratio=0.2):
    """
    Find the center time of the main activity burst
    using smoothed energy of |acc|.
    """
    secs = acc_df["seconds_elapsed"].astype(float).values
    x, y, z = acc_df["x"].values, acc_df["y"].values, acc_df["z"].values

    mag2 = x*x + y*y + z*z
    # use about 0.5 s worth of samples for smoothing
    n_env = max(3, int(round(env_win_s * 100)))  # assume ~100 Hz but only for smoothing width

    env = pd.Series(mag2).rolling(
        window=n_env, min_periods=max(3, int(0.4*n_env))
    ).mean().values

    idx_peak = int(np.nanargmax(env))
    peak_val = env[idx_peak] if np.isfinite(env[idx_peak]) else 0.0
    if peak_val <= 0:
        # fallback: center of whole recording
        return 0.5 * (secs.min() + secs.max())

    thr = peak_val * thresh_ratio
    iL = idx_peak
    while iL > 0 and (not np.isnan(env[iL]) and env[iL] > thr):
        iL -= 1
    iR = idx_peak
    while iR < len(env)-1 and (not np.isnan(env[iR]) and env[iR] > thr):
        iR += 1

    return 0.5 * (secs[iL] + secs[iR])


def slice_centered_window(df, center_time, n_points):
    """
    Slice exactly n_points rows centered on center_time.
    """
    secs = df["seconds_elapsed"].astype(float).values

    # index of sample closest to the desired center_time
    center_idx = int(np.argmin(np.abs(secs - center_time)))
    half = n_points // 2

    start_idx = center_idx - half
    end_idx = start_idx + n_points

    # keep indices inside valid range
    if start_idx < 0:
        start_idx = 0
        end_idx = n_points
    if end_idx > len(df):
        end_idx = len(df)
        start_idx = max(0, end_idx - n_points)

    # make sure they are integers
    start_idx = int(start_idx)
    end_idx   = int(end_idx)

    return df.iloc[start_idx:end_idx].copy()


def process_one_sample(sample_dir, out_dir, n_points, env_win_s=0.5, thresh_ratio=0.2):
    """
    Process one sample folder:
      - find activity center from accelerometer
      - cut fixed n_points window centered on it for all sensors
      - save results
    """
    acc_path  = os.path.join(sample_dir, "Accelerometer.csv")
    gyro_path = os.path.join(sample_dir, "Gyroscope.csv")
    grav_path = os.path.join(sample_dir, "Gravity.csv")

    if not (os.path.exists(acc_path) and os.path.exists(gyro_path) and os.path.exists(grav_path)):
        return {"sample_dir": sample_dir, "status": "skipped: missing csv"}

    acc  = pd.read_csv(acc_path)
    gyro = pd.read_csv(gyro_path)
    grav = pd.read_csv(grav_path)

    for df, name in [(acc, "Accelerometer"), (gyro, "Gyroscope"), (grav, "Gravity")]:
        for col in ["seconds_elapsed", "x", "y", "z"]:
            if col not in df.columns:
                return {"sample_dir": sample_dir, "status": f"skipped: bad {name}"}
        df.sort_values("seconds_elapsed", inplace=True)

    center_time = find_activity_center(acc, env_win_s=env_win_s, thresh_ratio=thresh_ratio)

    acc_win  = slice_centered_window(acc,  center_time, n_points)
    gyro_win = slice_centered_window(gyro, center_time, n_points)
    grav_win = slice_centered_window(grav, center_time, n_points)

    os.makedirs(out_dir, exist_ok=True)
    acc_win.to_csv(os.path.join(out_dir, "Accelerometer.csv"), index=False)
    gyro_win.to_csv(os.path.join(out_dir, "Gyroscope.csv"), index=False)
    grav_win.to_csv(os.path.join(out_dir, "Gravity.csv"), index=False)

    pd.DataFrame([{
        "source_dir": os.path.basename(sample_dir),
        "center_time": center_time,
        "points_saved": len(acc_win)
    }]).to_csv(os.path.join(out_dir, "window_info.csv"), index=False)

    status = "ok" if len(acc_win) == n_points else "ok (truncated)"
    return {"sample_dir": sample_dir, "out_dir": out_dir, "status": status}


def trim_sitdown_standup(raw_base, clean_base, categories=("sit_down", "stand_up"), window_size=3.0, verbose=True):
    """
    Process all samples in given categories.
    """
    n_points = window_size * 100 #fs: 100Hz
    results = []

    for cat in categories:
        raw_cat_dir   = os.path.join(raw_base, cat)
        clean_cat_dir = os.path.join(clean_base, cat)

        if not os.path.isdir(raw_cat_dir):
            if verbose:
                print(f"[WARN] missing folder: {cat}")
            continue

        for name in sorted(os.listdir(raw_cat_dir)):
            sample_dir = os.path.join(raw_cat_dir, name)
            if not os.path.isdir(sample_dir):
                continue

            sample_base = name.split("-")[0]
            out_dir = os.path.join(clean_cat_dir, sample_base)

            res = process_one_sample(sample_dir, out_dir, n_points)
            results.append(res)
            if verbose:
                pass
                #print(res)

    df = pd.DataFrame(results)
    if verbose and not df.empty:
        ok = (df["status"].str.startswith("ok")).sum()
        skipped = (df["status"].str.startswith("skipped")).sum()
        print(f"Done. Processed: {ok} | Skipped: {skipped}")

    return df


def stratified_person_label_split_from_csv(
    csv_path: str,
    test_ratio: float = 0.20,
    val_ratio_within_trainval: float = 0.20,
    random_state: int = 42
):
    '''
    Split dataset into train/val/test with stratification by class labels
    and grouping by person ID (so no person leaks across splits).

    Assumes dataset.csv format:
        - First column = label (y)
        - Columns 1..-2 = features (X)
        - Last column = folder/sample name (e.g., "liya_sample1")

    Splitting is done *within each (person, label)* bucket:
        1) train+val vs test (test_ratio, e.g. 0.20)
        2) train vs val on train+val (val_ratio_within_trainval, e.g. 0.20)
    => overall ~64/16/20 train/val/test.

    Parameters
    ----------
    csv_path : str
        Path to dataset.csv
    test_ratio : float, default=0.2
        Fraction of total data used for test
    val_ratio_within_trainval : float, default=0.2
        Fraction of train+val data used for validation
        (so final split is (1-test_ratio)*(1-val_ratio_within_trainval), (1-test_ratio)*val_ratio_within_trainval, test_ratio
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Row indices in the original CSV for each split.
    train_names, val_names, test_names : list[str]
        The last-column sample names belonging to each split (for debugging/logs).
    info : dict
        Metadata with counts per split and per person/label.

    '''
    rng = np.random.RandomState(random_state)

    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].values
    sample_names = df.iloc[:, -1].astype(str).values

    # Person parser: "liya_sample1" -> "liya"
    def _infer_person(name: str) -> str:
        if "_" in name:
            return name.split("_", 1)[0]
        m = re.match(r"^([A-Za-z]+)", name)
        return m.group(1) if m else name

    persons = np.array([_infer_person(s) for s in sample_names])

    # Build (person, label) -> list of row indices
    buckets = defaultdict(list)
    for i, (p, label) in enumerate(zip(persons, y)):
        buckets[(p, label)].append(i)

    # Allocate indices
    train_idx, val_idx, test_idx = [], [], []

    for (p, label), idx_list in buckets.items():
        # shuffle indices within each (person, label) bucket
        idx = np.array(idx_list)
        rng.shuffle(idx)

        n = len(idx)
        # Step 1: split test
        n_trainval = int(round((1.0 - test_ratio) * n))
        n_test = n - n_trainval

        idx_trainval = idx[:n_trainval]
        idx_test = idx[n_trainval:]

        # Step 2: split train vs val inside trainval
        n_train = int(round((1.0 - val_ratio_within_trainval) * len(idx_trainval)))
        n_val = len(idx_trainval) - n_train

        idx_train = idx_trainval[:n_train]
        idx_val = idx_trainval[n_train:]

        # collect
        train_idx.extend(idx_train.tolist())
        val_idx.extend(idx_val.tolist())
        test_idx.extend(idx_test.tolist())

    # Convert to arrays and (optionally) shuffle globally for randomness
    train_idx = np.array(train_idx, dtype=int)
    val_idx   = np.array(val_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Names for quick inspection
    train_names = sample_names[train_idx].tolist()
    val_names   = sample_names[val_idx].tolist()
    test_names  = sample_names[test_idx].tolist()

    # Build a small report
    def _counts(arr):
        return pd.Series(arr).value_counts().sort_index()

    info = {
        "sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "total": len(df),
        },
        "ratios": {
            "train": len(train_idx) / len(df),
            "val": len(val_idx) / len(df),
            "test": len(test_idx) / len(df),
        },
        "class_dist": {
            "train": _counts(y[train_idx]).to_dict(),
            "val": _counts(y[val_idx]).to_dict(),
            "test": _counts(y[test_idx]).to_dict(),
        },
        "person_dist": {
            "train": _counts(persons[train_idx]).to_dict(),
            "val": _counts(persons[val_idx]).to_dict(),
            "test": _counts(persons[test_idx]).to_dict(),
        },
    }

    return train_idx, val_idx, test_idx, train_names, val_names, test_names, info


def unsup_wrapper_select(X, model_class, model_kwargs,
                         n_features_to_select=15,
                         random_state=42,
                         mode="fixed"):
    """
    Forward selection wrapper for unsupervised clustering.
    Works with KMeans, GaussianMixture, and FuzzyCMeans.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    model_class : class
        Clustering algorithm class (e.g., KMeans, GaussianMixture).
    model_kwargs : dict
        Parameters passed to the clustering model.
    n_features_to_select : int
        Max number of features to try (in 'auto' mode, it's the limit).
    random_state : int
        Random seed (only used if model supports it).
    mode : str
        "fixed" → select exactly n_features_to_select features.
        "auto"  → stop early if silhouette stops improving.

    Returns
    -------
    X_best : pd.DataFrame
        Data restricted to best subset of features.
    best_subset : list
        List of selected feature names.
    best_score : float
        Best silhouette score.
    history : list of tuples
        (step, feature_added, silhouette_score) for each step.
    """
    features = list(X.columns)
    selected = []
    best_score = -1
    best_subset = []
    history = []

    for step in range(1, n_features_to_select + 1):
        scores = []
        for f in features:
            if f in selected:
                continue
            trial = selected + [f]

            # Safe init (some models don’t accept random_state)
            try:
                model = model_class(random_state=random_state, **model_kwargs)
            except TypeError:
                model = model_class(**model_kwargs)

            # Universal label extraction
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X[trial])
            elif hasattr(model, "predict"):
                model.fit(X[trial])
                labels = model.predict(X[trial])
            elif hasattr(model, "fit"):  # fuzzy C-means fallback
                model.fit(X[trial].to_numpy())
                labels = model.u.argmax(axis=1)
            else:
                raise ValueError(f"Model {model_class.__name__} not supported.")

            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(X[trial], labels)
            scores.append((score, f))

        if not scores:
            break

        # Pick best feature
        scores.sort(reverse=True, key=lambda x: x[0])
        best_feature, step_score = scores[0][1], scores[0][0]
        selected.append(best_feature)
        history.append((step, best_feature, step_score))

        print(f"Step {step} | Added: {best_feature} | silhouette={step_score:.3f}")

        # Track global best
        if step_score > best_score:
            best_score = step_score
            best_subset = selected.copy()
        elif mode == "auto":  # stop early
            print("No improvement, stopping early.")
            break

    return X[best_subset], best_subset, best_score, history
