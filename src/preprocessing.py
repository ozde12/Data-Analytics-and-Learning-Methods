import os
import numpy as np
import matplotlib.pyplot as plt

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


def process_folder(folder_data, activity, folder_name, out_root, window_size=5, overlap=0.5):
    """
    Detect trim start & end from Accelerometer.csv, create time windows,
    apply them to ALL CSVs in the folder, and save them.
    """
    df_acc = folder_data["Accelerometer.csv"]
    time = df_acc["seconds_elapsed"].values
    y_values = df_acc["y"].values

    # Parameters
    sample_rate = len(time) / (time[-1] - time[0])
    print(f"sample rate {sample_rate}")
    step = int(window_size * sample_rate)
    print(f"step {step}")
    stride = int(step * (1 - overlap))
    print(f"stride {stride}")

    # Use the updated trim detection
    trim_start, trim_end = detect_trim_bounds_rms_intersection(
        time,
        y_values,
        step,
        stride,
        frac_start=0.8,   # adjust to control when start is detected
        frac_end=0.7,     # adjust to control when end is detected
        smooth=5,
        debug=False
    )

    # Build time-based windows only between trim_start–trim_end
    windows = []
    start_idx = np.searchsorted(time, trim_start)
    end_idx   = np.searchsorted(time, trim_end)
    for i, start in enumerate(range(start_idx, end_idx - step + 1, stride)):
        end = start + step
        windows.append((time[start], time[end - 1]))

    # Save all files for each window
    out_dir = os.path.join(out_root, activity, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, (win_start, win_end) in enumerate(windows):
        win_dir = os.path.join(out_dir, f"window_{i+1}")
        os.makedirs(win_dir, exist_ok=True)
        for file_name, file_df in folder_data.items():
            if "seconds_elapsed" not in file_df.columns:
                continue
            mask = (file_df["seconds_elapsed"] >= win_start) & (file_df["seconds_elapsed"] <= win_end)
            file_df[mask].to_csv(os.path.join(win_dir, file_name), index=False)

    # Save summary plot (accelerometer only)
    plt.figure(figsize=(12, 5))
    plt.plot(time, df_acc["x"], label="x")
    plt.plot(time, df_acc["y"], label="y")
    plt.plot(time, df_acc["z"], label="z")

    # Mark trim start and end
    plt.axvline(trim_start, color="red", linestyle="--", label="Trim start")
    plt.axvline(trim_end, color="blue", linestyle="--", label="Trim end")

    # Shade windows
    for win_start, win_end in windows:
        plt.axvspan(win_start, win_end, color="gray", alpha=0.1)

    plt.title(f"{activity} - {folder_name} - Accelerometer.csv (Trim + Overlap)")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "summary_plot.png"))
    plt.close()

    return trim_start, trim_end, len(windows), out_dir

def detect_trim_bounds_rms_intersection(
    time,
    y_values,
    step,
    stride,
    frac_start: float = 0.2,   # fraction of peak for start
    frac_end: float = 0.2,     # fraction of peak for end
    smooth: int = 5,
    debug: bool = False
    ):

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

    # --- Debug / Summary plot ---
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
