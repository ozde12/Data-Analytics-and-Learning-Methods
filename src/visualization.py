import matplotlib.pyplot as plt
import pandas as pd 
import os

def plot_activity(activity_dir: str) -> None:
    """
    Plot accelerometer, gravity, and gyroscope data from a directory.

    :param activity_dir: Path to the activity session folder.
    :return: None (displays a 1x3 subplot figure).
    """

    sensors = ["Accelerometer.csv", "Gravity.csv", "Gyroscope.csv"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle(f"Sensor Data: {os.path.basename(activity_dir)}", fontsize=14)

    for ax, sensor in zip(axes, sensors):
        file_path = os.path.join(activity_dir, sensor)

        if not os.path.exists(file_path):
            ax.set_title(f"{sensor} (missing)")
            ax.axis("off")
            continue

        df = pd.read_csv(file_path)

        # --- use 'time' column and shift to start at 0 s ---
        if "time" in df.columns:
            t = df["time"].astype(float)
            # If the numbers look like nanoseconds since epoch, convert to seconds
            if t.max() > 1e12:        # heuristic threshold
                t = (t - t.iloc[0]) * 1e-9
            else:
                t = t - t.iloc[0]
        else:
            # fallback if 'time' missing
            t = df.index.astype(float)

        # --- plot x, y, z axes ---
        for axis_name in ["x", "y", "z"]:
            if axis_name in df.columns:
                ax.plot(t, df[axis_name], label=axis_name)

        ax.set_title(sensor.replace(".csv", ""))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_files(folders, folder_name, activity, files_to_plot):
    """
    Loop over multiple sensor files and plot x, y, z signals.

    Parameters
    ----------
    folders : dict
        Dictionary of {file_name: DataFrame} for a given folder.
    folder_name : str
        Name of the folder (e.g., running_1_xxx).
    activity : str
        Activity name (e.g., running, stand_up).
    files_to_plot : list[str]
        List of file names to plot (must exist in folders).
    """
    for file_name in files_to_plot:
        if file_name not in folders[folder_name]:
            print(f"Skipping {file_name} (not found in {folder_name})")
            continue

        df = folders[folder_name][file_name]
        time = df["seconds_elapsed"].values

        plt.figure(figsize=(12, 5))
        plt.plot(time, df["x"], label="x")
        plt.plot(time, df["y"], label="y")
        plt.plot(time, df["z"], label="z")

        plt.title(f"{activity} - {folder_name} - {file_name}")
        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

def plot_gravity(df_gravity, time, activity, folder_name):
    """
    Plot gravity sensor data.

    Parameters
    ----------
    df_gravity : DataFrame
        DataFrame containing gravity sensor data with columns 'x', 'y', 'z'.
    time : array-like
        Array of time values corresponding to the data points.
    activity : str
        Activity name (e.g., running, stand_up).
    folder_name : str
        Name of the folder (e.g., running_1_xxx).
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time, df_gravity["x"], label="x")
    plt.plot(time, df_gravity["y"], label="y")
    plt.plot(time, df_gravity["z"], label="z")

    plt.title(f"{activity} - {folder_name} - Gravity.csv")
    plt.xlabel("Time [s]")
    plt.ylabel("Gravity [m/s²]") 
    plt.legend()
    plt.show()

