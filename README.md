# Human Activity Recognition Project

This project builds a pipeline for processing smartphone sensor data to classify human activities such as sitting down, standing up, walking, running, and climbing stairs.  
The notebook provided can run end-to-end once the project directory downloaded and extracted from .zip.

---

## 📂 Project Structure

```
project_root/
│
├── data/
│   ├── raw/                # Raw sensor data collected from smartphones
│   └── trimmed_data/       # Trimmed data after removing inactive/noisy parts and windowing
│
├── group32_activity_recognition.ipynb    # Main notebook
├── group32_Report.pdf
├── README.md
├── requirements.txt                       # project dependencies
└── dataset.csv
```

---

## ⚙️ Setup Instructions

1. **Clone or Save the Project**
   - Download or clone the project to your local PC:
     ```bash
     git clone git@github.com:ozde12/Data-Analytics-and-Learning-Methods.git
     ```

2. **Install Dependencies**
   - Make sure you create a conda environment and install the required libraries:
     ```bash
    conda create -n .venv python=3.12 pip
    conda activate .venv
    pip install -r requirements.txt

     ```

3. **Run the Notebook**
   - Open the notebook: group32_activity_recognition.ipynb

   - Run all cells from start to finish.  
     The entire notebook is now able to execute end-to-end without manual intervention.

---

## 📊 Dataset Description

- **Raw Data:**  
  Collected using smartphone sensors (Accelerometer, Gyroscope, Gravity) at 100 Hz.  
  Stored as multiple CSV files per activity under `data/raw`.

- **Trimmed Data:**  
  Processed versions of raw recordings where inactive/noisy parts at the beginning and end have been removed and divided into windows.  
  Saved under `data/trimmed_data`.

- **Final Dataset (`dataset.csv`):**  
  Tabular dataset containing extracted features (statistical, time-domain, etc.) for each window of activity.  
  This is further processed before being an input for the machine learning models.

---

## 🚀 Notes
- Make sure the directory structure is preserved so the notebook can locate the data correctly.
- You can modify the notebook to experiment with new features or models.
- The default pipeline currently performs preprocessing, feature extraction, dataset creation, and model training in sequence.
