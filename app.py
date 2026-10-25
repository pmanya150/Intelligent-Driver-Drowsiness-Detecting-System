```python
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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin: 20px 0;
    }

    .alert-box {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }

    .drowsy-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
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

    st.error(
        "❌ drowsiness_model.joblib was not found."
    )

    st.info(
        "Place drowsiness_model.joblib in the same "
        "folder as app.py."
    )

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
# FEATURE NAMES
# =========================================================

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


# =========================================================
# PREPROCESS EEG SIGNAL
# =========================================================

def preprocess_channel(signal, fs):

    signal = np.asarray(
        signal,
        dtype=float
    )

    # Remove DC component
    signal = signal - np.mean(signal)

    # Band-pass filter
    low = 0.5
    high = 30

    nyquist = fs / 2

    b, a = butter(
        4,
        [
            low / nyquist,
            high / nyquist
        ],
        btype="band"
    )

    cleaned_signal = filtfilt(
        b,
        a,
        signal
    )

    return cleaned_signal


# =========================================================
# POWER SPECTRAL DENSITY
# =========================================================

def calculate_psd(window, fs):

    frequencies, psd = welch(
        window,
        fs=fs,
        nperseg=min(
            len(window),
            256
        )
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
        (frequencies >= low_freq)
        &
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
# EXTRACT 10 EEG FEATURES
# =========================================================

def extract_all_features(window, fs):

    features = {}

    powers = {}

    # -----------------------------------------------------
    # Absolute Band Powers
    # -----------------------------------------------------

    for band_name, (
        low,
        high
    ) in bands.items():

        power = band_power(
            window,
            fs,
            low,
            high
        )

        powers[band_name] = power

        features[band_name] = power


    # -----------------------------------------------------
    # Total Power
    # -----------------------------------------------------

    total_power = sum(
        powers.values()
    )

    if total_power == 0:

        total_power = 1e-10


    # -----------------------------------------------------
    # Relative Band Powers
    # -----------------------------------------------------

    for band_name in bands:

        features[
            band_name + "_relative"
        ] = (
            powers[band_name]
            /
            total_power
        )


    # -----------------------------------------------------
    # Theta / Alpha Ratio
    # -----------------------------------------------------

    features[
        "theta_alpha_ratio"
    ] = (
        powers["theta"]
        /
        (
            powers["alpha"]
            +
            1e-10
        )
    )


    # -----------------------------------------------------
    # Differential Entropy
    # -----------------------------------------------------

    variance = np.var(
        window
    )

    features[
        "differential_entropy"
    ] = (
        0.5
        *
        np.log(
            2
            *
            np.pi
            *
            np.e
            *
            variance
            +
            1e-10
        )
    )


    return features


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🚗 Intelligent Driver Drowsiness Detecting System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'EEG-Based Driver Vigilance Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ System Information"
)

st.sidebar.success(
    "Model Loaded Successfully"
)

st.sidebar.write(
    "**Sampling Frequency:** 128 Hz"
)

st.sidebar.write(
    "**Window Duration:** 2 seconds"
)

st.sidebar.write(
    "**Window Overlap:** 50%"
)

st.sidebar.write(
    "**EEG Features:** 10"
)

st.sidebar.write(
    "**Classifier:** Linear SVM"
)

st.sidebar.divider()

st.sidebar.info(
    "This is an academic research prototype "
    "for EEG-based driver drowsiness detection."
)


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

st.subheader(
    "📊 System Dashboard"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "EEG Samples",
        "2022"
    )

with col2:

    st.metric(
        "EEG Channels",
        "30"
    )

with col3:

    st.metric(
        "EEG Features",
        "10"
    )

with col4:

    st.metric(
        "Subjects",
        "11"
    )


st.info(
    "The system processes EEG signals, extracts "
    "frequency-domain features and predicts the "
    "driver's vigilance state using a trained "
    "machine learning model."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.divider()

st.subheader(
    "📂 EEG Dataset"
)

uploaded_file = st.file_uploader(
    "Upload EEG Dataset (.mat)",
    type=["mat"]
)


# =========================================================
# PROCESS UPLOADED DATA
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # LOAD MATLAB DATA
        # -------------------------------------------------

        data = loadmat(
            uploaded_file
        )


        st.success(
            "✅ EEG dataset uploaded successfully!"
        )


        # -------------------------------------------------
        # AVAILABLE VARIABLES
        # -------------------------------------------------

        available_keys = [
            key
            for key in data.keys()
            if not key.startswith("__")
        ]


        with st.expander(
            "View Dataset Variables"
        ):

            st.write(
                available_keys
            )


        # -------------------------------------------------
        # CHECK EEG DATA
        # -------------------------------------------------

        if "EEGsample" not in data:

            st.error(
                "❌ EEGsample variable was not "
                "found in the uploaded dataset."
            )

            st.stop()


        EEG = np.asarray(
            data["EEGsample"]
        )


        # -------------------------------------------------
        # DATASET INFORMATION
        # -------------------------------------------------

        st.subheader(
            "📋 Dataset Information"
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.write(
                f"**EEG Shape:** {EEG.shape}"
            )

        with info_col2:

            st.write(
                f"**Number of Samples:** {EEG.shape[0]}"
            )


        # =================================================
        # SAMPLE AND CHANNEL SELECTION
        # =================================================

        st.divider()

        st.subheader(
            "🔍 EEG Analysis"
        )


        selection_col1, selection_col2 = st.columns(2)


        with selection_col1:

            sample_number = st.number_input(
                "Select EEG Sample",
                min_value=0,
                max_value=EEG.shape[0] - 1,
                value=0,
                step=1
            )


        with selection_col2:

            channel_number = st.number_input(
                "Select EEG Channel",
                min_value=0,
                max_value=EEG.shape[1] - 1,
                value=0,
                step=1
            )


        st.caption(
            "Select an EEG sample and channel, "
            "then click Analyze EEG."
        )


        analyze_button = st.button(
            "🔍 Analyze EEG",
            use_container_width=True
        )


        # =================================================
        # ANALYSIS
        # =================================================

        if analyze_button:


            # -------------------------------------------------
            # GET SELECTED EEG SIGNAL
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
            # CREATE 2-SECOND WINDOWS
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
                len(cleaned_signal)
                -
                window_size
                +
                1,
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
                    "❌ EEG signal is too short "
                    "for a 2-second window."
                )

                st.stop()


            # =================================================
            # FEATURE EXTRACTION
            # =================================================

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

                    feature_dict[
                        "delta_relative"
                    ],

                    feature_dict[
                        "theta_relative"
