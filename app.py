import streamlit as st
import joblib
import numpy as np
import os

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intelligent Driver Drowsiness Detecting System",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "drowsiness_model.joblib"
)


# =========================================================
# LOAD MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error("drowsiness_model.joblib not found!")
    st.stop()

model = joblib.load(MODEL_PATH)


# =========================================================
# EEG SETTINGS
# =========================================================

fs = 128

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_channel(signal, fs):

    signal = np.asarray(signal, dtype=float)

    # Remove DC component
    signal = signal - np.mean(signal)

    # Band-pass filter
    low = 0.5
    high = 30

    nyquist = fs / 2

    b, a = butter(
        4,
        [low / nyquist, high / nyquist],
        btype="band"
    )

    cleaned_signal = filtfilt(
        b,
        a,
        signal
    )

    return cleaned_signal


# =========================================================
# PSD
# =========================================================

def calculate_psd(window, fs):

    frequencies, psd = welch(
        window,
        fs=fs,
        nperseg=min(len(window), 256)
    )

    return frequencies, psd


# =========================================================
# BAND POWER
# =========================================================

def band_power(
    window,
    fs,
    low_freq,
    high_freq
):

    frequencies, psd = calculate_psd(
        window,
        fs
    )

    mask = (
        (frequencies >= low_freq) &
        (frequencies <= high_freq)
    )

    if np.sum(mask) == 0:
        return 0.0

    power = np.trapezoid(
        psd[mask],
        frequencies[mask]
    )

    return power


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_all_features(window, fs):

    features = {}
    powers = {}

    # Absolute band powers
    for band_name, (low, high) in bands.items():

        power = band_power(
            window,
            fs,
            low,
            high
        )

        powers[band_name] = power
        features[band_name] = power

    # Total power
    total_power = sum(
        powers.values()
    )

    if total_power == 0:
        total_power = 1e-10

    # Relative band powers
    for band_name in bands:

        features[
            band_name + "_relative"
        ] = (
            powers[band_name] /
            total_power
        )

    # Theta / Alpha ratio
    features["theta_alpha_ratio"] = (
        powers["theta"] /
        (powers["alpha"] + 1e-10)
    )

    # Differential entropy
    variance = np.var(window)

    features["differential_entropy"] = (
        0.5 *
        np.log(
            2 *
            np.pi *
            np.e *
            variance +
            1e-10
        )
    )

    return features


# =========================================================
# HEADER
# =========================================================

st.title(
    "🚗 Intelligent Driver Drowsiness Detecting System"
)

st.subheader(
    "EEG-Based Driver Vigilance Monitoring"
)

st.write(
    "Upload an EEG dataset and analyze the signal "
    "using the trained machine learning model."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ System Information")

st.sidebar.write(
    "Sampling Frequency: **128 Hz**"
)

st.sidebar.write(
    "Window Duration: **2 seconds**"
)

st.sidebar.write(
    "Overlap: **50%**"
)

st.sidebar.write(
    "Features: **10**"
)

st.sidebar.write(
    "Classifier: **Linear SVM**"
)


# =========================================================
# MODEL STATUS
# =========================================================

st.sidebar.success(
    "Model Loaded Successfully"
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload EEG Dataset (.mat)",
    type=["mat"]
)


# =========================================================
# PROCESS DATASET
# =========================================================

if uploaded_file is not None:

    try:

        # Load MATLAB file
        data = loadmat(
            uploaded_file
        )

        st.success(
            "EEG dataset uploaded successfully!"
        )

        # -------------------------------------------------
        # DATASET VARIABLES
        # -------------------------------------------------

        available_keys = [
            key
            for key in data.keys()
            if not key.startswith("__")
        ]

        st.subheader(
            "📂 Dataset Information"
        )

        st.write(
            "Available variables:"
        )

        st.write(
            available_keys
        )

        # -------------------------------------------------
        # CHECK EEG DATA
        # -------------------------------------------------

        if "EEGsample" not in data:

            st.error(
                "EEGsample variable was not found "
                "in the uploaded dataset."
            )

            st.stop()

        EEG = np.asarray(
            data["EEGsample"]
        )

        # -------------------------------------------------
        # DATASET SHAPE
        # -------------------------------------------------

        st.info(
            f"EEG Shape: {EEG.shape}"
        )

        # -------------------------------------------------
        # SAMPLE SELECTION
        # -------------------------------------------------

        sample_number = st.number_input(
            "Select EEG Sample",
            min_value=0,
            max_value=EEG.shape[0] - 1,
            value=0,
            step=1
        )

        # -------------------------------------------------
        # CHANNEL SELECTION
        # -------------------------------------------------

        channel_number = st.number_input(
            "Select EEG Channel",
            min_value=0,
            max_value=EEG.shape[1] - 1,
            value=0,
            step=1
        )

        # -------------------------------------------------
        # ANALYZE BUTTON
        # -------------------------------------------------

        analyze_button = st.button(
            "🔍 Analyze EEG",
            use_container_width=True
        )

        if analyze_button:

            # -------------------------------------------------
            # GET EEG SIGNAL
            # -------------------------------------------------

            signal = EEG[
                sample_number,
                channel_number,
                :
            ]

            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------

            cleaned_signal = preprocess_channel(
                signal,
                fs
            )

            # -------------------------------------------------
            # CREATE WINDOWS
            # -------------------------------------------------

            window_size = int(
                2 * fs
            )

            stride = int(
                window_size * 0.5
            )

            windows = []

            for start in range(
                0,
                len(cleaned_signal) -
                window_size + 1,
                stride
            ):

                window = cleaned_signal[
                    start:
                    start + window_size
                ]

                windows.append(
                    window
                )

            windows = np.asarray(
                windows
            )

            # -------------------------------------------------
            # CHECK WINDOWS
            # -------------------------------------------------

            if len(windows) == 0:

                st.error(
                    "Signal is too short for "
                    "a 2-second window."
                )

                st.stop()

            # -------------------------------------------------
            # FEATURE EXTRACTION
            # -------------------------------------------------

            feature_rows = []

            for window in windows:

                feature_dict = (
                    extract_all_features(
                        window,
                        fs
                    )
                )

                feature_rows.append([
                    feature_dict["delta"],
                    feature_dict["theta"],
                    feature_dict["alpha"],
                    feature_dict["beta"],
                    feature_dict["delta_relative"],
                    feature_dict["theta_relative"],
                    feature_dict["alpha_relative"],
                    feature_dict["beta_relative"],
                    feature_dict["theta_alpha_ratio"],
                    feature_dict["differential_entropy"]
                ])

            X_input = np.asarray(
                feature_rows
            )

            # -------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------

            predictions = model.predict(
                X_input
            )

            # -------------------------------------------------
            # MAJORITY VOTING
            # -------------------------------------------------

            final_prediction = int(
                np.bincount(
                    predictions
                ).argmax()
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "🧠 Drowsiness Prediction"
            )

            if final_prediction == 1:

                st.error(
                    "⚠️ DROWSY — "
                    "Driver drowsiness detected"
                )

                st.warning(
                    "Please consider taking a break."
                )

            else:

                st.success(
                    "✅ ALERT — "
                    "Driver appears alert"
                )

            # -------------------------------------------------
            # RESULT METRICS
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Prediction",
                    "Drowsy"
                    if final_prediction == 1
                    else "Alert"
                )

            with col2:

                st.metric(
                    "Windows Analyzed",
                    len(windows)
                )

            with col3:

                st.metric(
                    "EEG Channel",
                    channel_number
                )

            # -------------------------------------------------
            # FEATURE TABLE
            # -------------------------------------------------

            st.subheader(
                "📊 Extracted EEG Features"
            )

            feature_names = [
                "delta",
                "theta",
                "alpha",
                "beta",
                "delta_relative",
                "theta_relative",
                "alpha_relative",
                "beta_relative",
                "theta_alpha_ratio",
                "differential_entropy"
            ]

            feature_table = {
                "Feature": feature_names,
                "Value": X_input[0]
            }

            st.dataframe(
                feature_table,
                use_container_width=True
            )

            # -------------------------------------------------
            # EEG SIGNAL
            # -------------------------------------------------

            st.subheader(
                "📈 Preprocessed EEG Signal"
            )

            st.line_chart(
                cleaned_signal
            )

            # -------------------------------------------------
            # PREDICTION DISTRIBUTION
            # -------------------------------------------------

            st.subheader(
                "📊 Window Prediction Distribution"
            )

            alert_count = int(
                np.sum(predictions == 0)
            )

            drowsy_count = int(
                np.sum(predictions == 1)
            )

            prediction_distribution = {
                "Alert": alert_count,
                "Drowsy": drowsy_count
            }

            st.bar_chart(
                prediction_distribution
            )

    except Exception as e:

        st.error(
            f"Error while processing EEG dataset: {e}"
        )