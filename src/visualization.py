import matplotlib.pyplot as plt

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
