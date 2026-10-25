# Intelligent Driver Drowsiness Detecting System

## 📌 Project Overview

**Intelligent Driver Drowsiness Detecting System** is an EEG-based machine learning system designed to detect driver vigilance states and identify possible drowsiness.

The system processes EEG signals, performs preprocessing, divides the signals into time windows, extracts meaningful EEG features, and uses machine learning models to classify the driver's state as **Alert** or **Drowsy**.

A Streamlit-based user interface is also provided to demonstrate the prediction pipeline.

---

## 🎯 Objectives

- Process and preprocess EEG signals for driver vigilance analysis.
- Divide EEG signals into fixed-length overlapping windows.
- Extract frequency-domain and statistical EEG features.
- Classify driver states using machine learning models.
- Evaluate the system using different validation strategies.
- Perform subject-independent evaluation using LOSO Cross-Validation.
- Provide a user-friendly interface for EEG-based prediction.

---

## 🧠 System Workflow

```text
EEG Dataset
     ↓
EEG Signal Preprocessing
     ↓
2-Second Windowing
     ↓
EEG Feature Extraction
     ↓
Feature Scaling
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Majority Voting
     ↓
Alert / Drowsy
```

---

## 📊 Dataset

The project uses EEG data stored in MATLAB `.mat` format.

The processed EEG dataset contains:

- **2022 original EEG samples**
- **30 EEG channels**
- **384 data points per channel**
- **11 subjects**
- **2 states**
  - `0` → Alert
  - `1` → Drowsy

### Dataset Distribution

| State | Samples |
|------|--------:|
| Alert | 1011 |
| Drowsy | 1011 |
| **Total** | **2022** |

The original EEG dataset is **not included in this repository** because of its large file size.

---

## 🔬 EEG Preprocessing

The EEG signals are preprocessed before feature extraction.

The preprocessing pipeline includes:

- Removal of the signal mean/DC component.
- Band-pass filtering.
- Frequency analysis using Power Spectral Density (PSD).
- Segmentation into 2-second windows.
- 50% overlap between consecutive windows.

---

## 📈 Extracted Features

The system extracts **10 features** from each EEG window.

| Feature | Description |
|---|---|
| Delta | Delta-band power |
| Theta | Theta-band power |
| Alpha | Alpha-band power |
| Beta | Beta-band power |
| Delta Relative | Relative delta power |
| Theta Relative | Relative theta power |
| Alpha Relative | Relative alpha power |
| Beta Relative | Relative beta power |
| Theta/Alpha Ratio | Ratio between theta and alpha power |
| Differential Entropy | Statistical EEG feature |

---

## 🤖 Machine Learning Models

Two machine learning models were evaluated during the project:

### 1. RBF Support Vector Machine

Accuracy:

**69.47%**

### 2. Random Forest

Accuracy:

**67.02%**

A Linear SVM was additionally evaluated using subject-independent strategies.

---

## 📋 Model Evaluation

| Evaluation Method | Model | Accuracy |
|---|---|---:|
| Random Sample Split | RBF SVM | **69.47%** |
| Random Sample Split | Random Forest | **67.02%** |
| Subject-wise Split | Linear SVM | **70.62%** |
| LOSO Cross-Validation | Linear SVM | **64.08%** |
| LOSO + Majority Voting | Linear SVM | **69.44%** |

### LOSO Performance

The Leave-One-Subject-Out (LOSO) evaluation was performed across all 11 subjects.

- **Mean Accuracy:** 64.08%
- **Best Subject Accuracy:** 78.38%
- **Lowest Subject Accuracy:** 51.77%
- **Standard Deviation:** 8.15%

The LOSO evaluation provides a more subject-independent assessment of the model.

---

## 🖥️ User Interface

The project includes a **Streamlit-based interface**.

The UI allows the user to:

1. Upload an EEG `.mat` dataset.
2. Select an EEG sample.
3. Select an EEG channel.
4. Preprocess the selected signal.
5. Generate EEG windows.
6. Extract the 10 EEG features.
7. Use the trained SVM model for prediction.
8. Apply majority voting across windows.
9. Display the final result as:
   - **Alert**
   - **Drowsy**
10. Visualize the processed EEG signal and extracted features.

---

## 🛠️ Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **SciPy**
- **Scikit-learn**
- **Matplotlib**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**

---

## 📁 Project Structure

```text
Intelligent-Driver-Drowsiness-Detection/
│
├── app.py
├── drowsiness_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   └── driver_drowsiness.ipynb
│
└── data/
    └── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Intelligent-Driver-Drowsiness-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

Upload the required EEG `.mat` dataset through the interface and select a sample and channel for analysis.

---

## 📂 Dataset Setup

Because the EEG dataset is large, it is **not stored in this GitHub repository**.

After obtaining the dataset, keep it locally and upload it through the Streamlit interface.

The expected MATLAB dataset should contain the EEG data variable used by the application.

---

## 📊 Results

The system achieved approximately **69.44% sample-level accuracy** using LOSO predictions followed by majority voting over the original EEG samples.

### Sample-Level Classification

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Alert | 0.6870 | 0.7141 | 0.7003 |
| Drowsy | 0.7024 | 0.6746 | 0.6882 |

**Sample-Level Accuracy: 69.44%**

---

## 🔮 Future Scope

Possible future improvements include:

- Integration of additional modalities such as facial/eye-based signals.
- Lightweight deep learning models.
- Temporal sequence modelling.
- Real-time EEG stream processing.
- Visual and audible drowsiness alerts.
- Further optimization for real-time deployment.
- Evaluation on larger and more diverse datasets.

---

## ⚠️ Disclaimer

This project is a **research and academic prototype**. It should not be used as a safety-critical system for real-world driving without extensive additional validation.

---

## 👩‍💻 Project

**Intelligent Driver Drowsiness Detecting System**

Developed as an academic/research project using EEG signal processing and machine learning.
