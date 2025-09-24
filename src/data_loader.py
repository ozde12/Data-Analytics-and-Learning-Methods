import os
import pandas as pd

def load_all_data():
    home = os.path.expanduser("~")
    base_dir = os.path.join(
        home,
        "Documents",
        "data analytics",
        "Data-Analytics-and-Learning-Methods",
        "data",
        "raw",
    )

    activities = ["running", "sit_down", "stairs", "stand_up", "walking"]

    data = {}

    for activity in activities:
        activity_dir = os.path.join(base_dir, activity)

        subfolders = [
            os.path.join(activity_dir, d)
            for d in os.listdir(activity_dir)
            if os.path.isdir(os.path.join(activity_dir, d))
        ]

        data[activity] = {}

        for folder in subfolders:
            folder_name = os.path.basename(folder)
            csvs = {}
            for file_name in os.listdir(folder):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(folder, file_name)
                    print(f"Loading {file_path}...")
                    try:
                        df = pd.read_csv(file_path, skipinitialspace=True)
                        if df.empty:
                            continue
                        df.columns = df.columns.str.strip()
                        csvs[file_name] = df
                    except Exception as e:
                        print(f"Could not load {file_name}: {e}")
            if csvs:
                data[activity][folder_name] = csvs

    return data