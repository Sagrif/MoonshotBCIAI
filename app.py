import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

# Import custom modules
from data_handler import load_eeg_data, preprocess_data
from feature_extraction import extract_features
from model_training import train_model, evaluate_model
from visualization import plot_eeg_data, plot_features, plot_brain_activity, plot_confusion_matrix, plot_roc_curve
from utils import create_example_description, create_demo_data

# Set page configuration
st.set_page_config(
    page_title="BCI Epilepsy Detection",
    page_icon="🧠",
    layout="wide"
)

# Application title and description
st.title("BCI Epilepsy Detection System")
st.markdown("""
This application analyzes Brain-Computer Interface (BCI) data using machine learning to detect epilepsy patterns.
Upload your EEG data to get started with the analysis.
""")

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Navigation functions
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Sidebar for navigation and progress tracking
st.sidebar.title("Progress")

# Define workflow stages and requirements
workflow_stages = [
    {"name": "Home", "description": "Introduction", "enabled": True},
    {"name": "Data Upload", "description": "Upload EEG Data", "enabled": True},
    {"name": "Preprocessing", "description": "Clean & Filter Data", "enabled": st.session_state.data is not None},
    {"name": "Feature Extraction", "description": "Extract Features", "enabled": st.session_state.preprocessed_data is not None},
    {"name": "Model Training", "description": "Train ML Model", "enabled": st.session_state.features is not None},
    {"name": "Results", "description": "View Results", "enabled": st.session_state.model is not None}
]

# Create progress indicator and navigation
current_stage_idx = [i for i, stage in enumerate(workflow_stages) if stage["name"] == st.session_state.page][0]
progress_percentage = (current_stage_idx) / (len(workflow_stages) - 1) * 100 if len(workflow_stages) > 1 else 0

st.sidebar.progress(int(progress_percentage))
st.sidebar.markdown(f"**Current Stage:** {workflow_stages[current_stage_idx]['name']}")

# Custom CSS for navigation buttons and UI elements with neobrutalism style
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;900&family=Work+Sans:wght@400;600;700&display=swap');
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Work Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Header fonts */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Improve paragraph readability */
    p, li, .stMarkdown p, .stMarkdown li {
        line-height: 1.6 !important;
        font-size: 1.05rem !important;
    }
    
    /* Neobrutalism button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0px !important;
        text-align: left;
        border: 4px solid black !important;
        box-shadow: 4px 4px 0px black !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 4px 0px !important;
        transition: all 0.2s ease;
    }
    
    /* Dark mode adjustments */
    .stApp.stApp {
        --text-color: white;
        --background-color: #111;
    }
    
    /* Light mode adjustments */
    @media (prefers-color-scheme: light) {
        .stApp.stApp {
            --text-color: black;
            --background-color: white;
        }
    }
    
    /* Hover effects */
    .stButton > button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px black !important;
    }
    
    .stButton > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px black !important;
    }
    
    /* Primary button styling with neobrutalism */
    button[kind="primary"] {
        background-color: #2e6fba !important;
        color: white !important;
        border: 4px solid black !important;
        box-shadow: 4px 4px 0px black !important;
    }
    
    /* Make progress bar use neutral colors instead of red */
    .stProgress > div > div {
        background-color: #2e6fba !important;
    }
    
    /* Adjust other red UI elements */
    .css-zt5igj {
        border-left-color: #2e6fba !important;
    }
    
    /* Custom button styling for the header */
    .stApp .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        z-index: 999;
        display: flex;
        justify-content: center;
        pointer-events: none; /* Let clicks go through to elements below */
    }
    
    .stApp .fixed-header button {
        pointer-events: auto; /* Make the button clickable */
        background-color: #ffde59 !important;
        color: black !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 0 !important;
        border: 4px solid black !important;
        box-shadow: 4px 4px 0px black !important;
        text-transform: uppercase !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        letter-spacing: 1px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .stApp .fixed-header button:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 6px 6px 0px black !important;
    }
    
    /* Add padding to the top of the page to avoid content being hidden under the fixed header */
    .main .block-container {
        padding-top: 6rem !important;
    }
    
    /* Style for sidebar */
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
        border-right: 4px solid black;
    }
    
    /* Style for code blocks */
    code {
        font-family: 'Space Mono', monospace !important;
        background-color: #f1f1f1;
        padding: 2px 5px;
        border-radius: 0 !important;
        border: 2px solid black !important;
        font-size: 0.9em !important;
    }
    
    /* Style for tables */
    table {
        border: 4px solid black !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
    }
    
    th, td {
        border: 2px solid black !important;
        padding: 8px 12px !important;
    }
    
    th {
        background-color: #ffde59 !important;
        color: black !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Display workflow stages with status
for i, stage in enumerate(workflow_stages):
    if i == current_stage_idx:
        st.sidebar.markdown(f"**{stage['name']}**: {stage['description']} (Current)")
    elif stage["enabled"]:
        if st.sidebar.button(f"{stage['name']}: {stage['description']}", key=f"nav_{i}"):
            go_to_page(stage["name"])
    else:
        st.sidebar.markdown(f"<span style='color:gray'>{stage['name']}: {stage['description']}</span>", unsafe_allow_html=True)

# Use the session state for page selection
page = st.session_state.page

# Only show the fixed header with skip button on the Home page
if page == "Home":
    # Create a container at the top of the page for the fixed header
    header_container = st.container()
    
    with header_container:
        # Apply the fixed header CSS
        st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
        
        # Add the Skip to Data Upload button
        if st.button("Skip to Data Upload", key="skip_to_data_upload_fixed"):
            go_to_page("Data Upload")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Home page with information
if page == "Home":
    # Use a simpler approach for the skip button
    # The floating "Skip to Data Upload" button is already added via CSS
    
    st.header("Understanding Epilepsy and EEG Data")
    
    st.markdown("""
    ## What is Epilepsy?
    Epilepsy is a neurological disorder characterized by recurrent seizures. These seizures are caused by abnormal electrical activity in the brain, which can be detected using electroencephalography (EEG).

    ## How BCI/EEG Data Helps in Epilepsy Detection
    Brain-Computer Interfaces (BCI) and EEG recordings capture brain electrical activity, allowing us to identify patterns associated with epilepsy:
    
    1. **Epileptiform Discharges**: Sharp waves, spikes, or spike-wave complexes
    2. **Abnormal Rhythmic Activity**: Especially in the theta and delta frequency bands
    3. **Changes Before, During, and After Seizures**: Pre-ictal, ictal, and post-ictal patterns
    
    ## How This Application Works
    1. **Data Upload**: Upload your EEG/BCI data in standard formats
    2. **Preprocessing**: Clean and filter the EEG signals
    3. **Feature Extraction**: Extract time and frequency domain features
    4. **Model Training**: Train a machine learning model to detect epilepsy patterns
    5. **Results Visualization**: View performance metrics and classification results
    """)
    
    # Display example description
    create_example_description()

# Data Upload page
elif page == "Data Upload":
    st.header("Upload or Generate EEG/BCI Data")
    
    # Create tabs for uploading real data or using sample data
    upload_tab, sample_tab = st.tabs(["Upload Real Data", "Use Sample Data"])
    
    with upload_tab:
        st.markdown("""
        Upload your EEG/BCI data file. The following formats are supported:
        - European Data Format (.edf)
        - BrainVision (.vhdr, .vmrk, .eeg)
        - Biosemi (.bdf)
        - EEGLab (.set)
        - Neuroscan (.cnt)
        - Plain text files (.txt, .csv) with proper formatting
        """)
        
        uploaded_file = st.file_uploader("Choose a file", type=["edf", "bdf", "vhdr", "set", "cnt", "txt", "csv"])
        
        if uploaded_file is not None:
            # Display file details
            file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
            st.write(file_details)
            
            # Save the uploaded file temporarily
            with st.spinner('Loading data...'):
                # Create temp directory if it doesn't exist
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                temp_path = f"temp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Load the data
                    data, sfreq, ch_names = load_eeg_data(temp_path)
                    
                    if data is not None:
                        st.session_state.data = {
                            'raw_data': data,
                            'sfreq': sfreq,
                            'ch_names': ch_names,
                            'filepath': temp_path,
                            'data_type': 'uploaded'
                        }
                        st.session_state.file_uploaded = True
                        
                        # Display basic information about the data
                        st.success(f"Successfully loaded data with {len(ch_names)} channels at {sfreq}Hz sampling rate")
                        
                        # Display a preview of the EEG data
                        st.subheader("EEG Data Preview")
                        fig = plot_eeg_data(data, sfreq, ch_names, duration=5, n_channels=min(5, len(ch_names)))
                        st.pyplot(fig)
                    else:
                        st.error("Failed to load the data. Please check if the file format is supported.")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    with sample_tab:
        st.markdown("""
        Generate sample EEG data for testing the epilepsy detection pipeline. You can customize the 
        parameters of the sample data below.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_channels = st.slider("Number of Channels", min_value=1, max_value=32, value=19, step=1)
            duration = st.slider("Duration (seconds)", min_value=5, max_value=60, value=30, step=5)
        
        with col2:
            sfreq = st.slider("Sampling Frequency (Hz)", min_value=100, max_value=1000, value=250, step=50)
            include_seizure = st.checkbox("Include simulated seizure patterns", value=True)
        
        if st.button("Generate Sample Data"):
            with st.spinner('Generating sample data...'):
                try:
                    # Calculate number of samples
                    n_samples = int(duration * sfreq)
                    
                    # Generate the sample data using the utility function
                    data, ch_names = create_demo_data(n_samples=n_samples, sfreq=sfreq, n_channels=n_channels, 
                                                      include_seizure=include_seizure)
                    
                    # Store in session state
                    st.session_state.data = {
                        'raw_data': data,
                        'sfreq': sfreq,
                        'ch_names': ch_names,
                        'filepath': 'sample_data',
                        'data_type': 'sample'
                    }
                    st.session_state.file_uploaded = True
                    
                    # Display basic information about the data
                    st.success(f"Successfully generated sample data with {len(ch_names)} channels at {sfreq}Hz sampling rate")
                    
                    # Display a preview of the EEG data
                    st.subheader("EEG Data Preview")
                    fig = plot_eeg_data(data, sfreq, ch_names, duration=5, n_channels=min(5, len(ch_names)))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating sample data: {str(e)}")
    
    # Shared section for both tabs
    if st.session_state.file_uploaded:
        st.info(f"Current data: {'Sample generated data' if st.session_state.data.get('data_type') == 'sample' else os.path.basename(st.session_state.data['filepath'])}")
        
        # Option to view the current data
        if st.button("View Current Data"):
            fig = plot_eeg_data(
                st.session_state.data['raw_data'], 
                st.session_state.data['sfreq'], 
                st.session_state.data['ch_names'], 
                duration=5, 
                n_channels=min(5, len(st.session_state.data['ch_names']))
            )
            st.pyplot(fig)
        
        # Option to continue to preprocessing
        if st.button("Next: Preprocessing →"):
            go_to_page("Preprocessing")

# Preprocessing page
elif page == "Preprocessing":
    st.header("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("No data uploaded. Please upload data first.")
        if st.button("← Back to Data Upload"):
            go_to_page("Data Upload")
    else:
        st.markdown("""
        Preprocess the EEG data to improve signal quality and prepare it for feature extraction.
        The following preprocessing steps can be applied:
        """)
        
        with st.form("preprocessing_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                apply_notch = st.checkbox("Apply Notch Filter (remove power line noise)", True)
                notch_freq = st.number_input("Notch Filter Frequency (Hz)", min_value=50, max_value=60, value=50, step=10)
                
                apply_bandpass = st.checkbox("Apply Bandpass Filter", True)
                bandpass_low = st.number_input("Bandpass Low Cutoff (Hz)", min_value=0.1, max_value=40.0, value=1.0, step=0.5)
                bandpass_high = st.number_input("Bandpass High Cutoff (Hz)", min_value=10.0, max_value=200.0, value=70.0, step=5.0)
            
            with col2:
                apply_reference = st.checkbox("Re-reference Data", True)
                reference_type = st.selectbox("Reference Type", ["Average", "Mastoids"])
                
                remove_artifacts = st.checkbox("Remove Artifacts", True)
                artifact_method = st.selectbox("Artifact Removal Method", ["ICA", "Threshold"])
                
                resample = st.checkbox("Resample Data", False)
                resample_freq = st.number_input("Resampling Frequency (Hz)", min_value=100, max_value=1000, value=250, step=50)
            
            submit_button = st.form_submit_button("Apply Preprocessing")
            
            if submit_button:
                with st.spinner('Preprocessing data...'):
                    try:
                        # Get preprocessing parameters
                        preproc_params = {
                            'apply_notch': apply_notch,
                            'notch_freq': notch_freq,
                            'apply_bandpass': apply_bandpass,
                            'bandpass_low': bandpass_low,
                            'bandpass_high': bandpass_high,
                            'apply_reference': apply_reference,
                            'reference_type': reference_type,
                            'remove_artifacts': remove_artifacts,
                            'artifact_method': artifact_method,
                            'resample': resample,
                            'resample_freq': resample_freq if resample else None
                        }
                        
                        # Apply preprocessing
                        preprocessed_data = preprocess_data(
                            st.session_state.data['raw_data'],
                            st.session_state.data['sfreq'],
                            st.session_state.data['ch_names'],
                            preproc_params
                        )
                        
                        if preprocessed_data is not None:
                            st.session_state.preprocessed_data = preprocessed_data
                            st.success("Preprocessing completed successfully")
                            
                            # Display before & after plots
                            st.subheader("Before and After Preprocessing")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Before Preprocessing**")
                                fig = plot_eeg_data(
                                    st.session_state.data['raw_data'],
                                    st.session_state.data['sfreq'],
                                    st.session_state.data['ch_names'],
                                    duration=5,
                                    n_channels=min(3, len(st.session_state.data['ch_names']))
                                )
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("**After Preprocessing**")
                                fig = plot_eeg_data(
                                    preprocessed_data['data'],
                                    preprocessed_data['sfreq'],
                                    preprocessed_data['ch_names'],
                                    duration=5,
                                    n_channels=min(3, len(preprocessed_data['ch_names']))
                                )
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during preprocessing: {str(e)}")
        
        # Display current preprocessed data if it exists
        if st.session_state.preprocessed_data is not None:
            st.subheader("Current Preprocessed Data")
            st.write(f"Sampling rate: {st.session_state.preprocessed_data['sfreq']} Hz")
            st.write(f"Channels: {len(st.session_state.preprocessed_data['ch_names'])}")
            
            # Option to continue to feature extraction
            if st.button("Next: Feature Extraction →"):
                go_to_page("Feature Extraction")

# Feature Extraction page
elif page == "Feature Extraction":
    st.header("Feature Extraction")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No preprocessed data available. Please complete preprocessing first.")
        if st.button("← Back to Preprocessing"):
            go_to_page("Preprocessing")
    else:
        st.markdown("""
        Extract features from the preprocessed EEG data for epilepsy detection.
        Select the features you want to extract:
        """)
        
        with st.form("feature_extraction_form"):
            # Time domain features
            st.subheader("Time Domain Features")
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                extract_mean = st.checkbox("Mean", True)
                extract_variance = st.checkbox("Variance", True)
                extract_skewness = st.checkbox("Skewness", True)
            with time_col2:
                extract_kurtosis = st.checkbox("Kurtosis", True)
                extract_line_length = st.checkbox("Line Length", True)
                extract_hjorth = st.checkbox("Hjorth Parameters", True)
            
            # Frequency domain features
            st.subheader("Frequency Domain Features")
            freq_col1, freq_col2 = st.columns(2)
            with freq_col1:
                extract_bands = st.checkbox("Frequency Bands Power", True)
                extract_spectral_entropy = st.checkbox("Spectral Entropy", True)
            with freq_col2:
                extract_psd = st.checkbox("Power Spectral Density", True)
                extract_coherence = st.checkbox("Coherence", False)
            
            # Time-frequency domain features
            st.subheader("Time-Frequency Domain Features")
            tf_col1, tf_col2 = st.columns(2)
            with tf_col1:
                extract_wavelet = st.checkbox("Wavelet Coefficients", True)
            with tf_col2:
                extract_hilbert = st.checkbox("Hilbert Transform", False)
            
            # Other parameters
            st.subheader("Feature Extraction Parameters")
            window_size = st.slider("Window Size (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            window_overlap = st.slider("Window Overlap (%)", min_value=0, max_value=90, value=50, step=10)
            
            # Create segmentation for known seizure periods
            st.subheader("Seizure Annotation")
            st.markdown("If you know the time periods of seizures in your data, you can annotate them here:")
            
            has_seizure_info = st.checkbox("I have seizure timing information")
            
            if has_seizure_info:
                num_seizures = st.number_input("Number of seizure periods", min_value=1, max_value=10, value=1)
                
                seizure_times = []
                for i in range(num_seizures):
                    st.markdown(f"**Seizure {i+1}**")
                    sei_col1, sei_col2 = st.columns(2)
                    with sei_col1:
                        start_time = st.number_input(f"Start time (seconds) for seizure {i+1}", 
                                                  min_value=0.0, 
                                                  max_value=float(len(st.session_state.preprocessed_data['data'][0]) / 
                                                                st.session_state.preprocessed_data['sfreq']), 
                                                  value=0.0)
                    with sei_col2:
                        end_time = st.number_input(f"End time (seconds) for seizure {i+1}", 
                                                min_value=0.0, 
                                                max_value=float(len(st.session_state.preprocessed_data['data'][0]) / 
                                                              st.session_state.preprocessed_data['sfreq']), 
                                                value=min(30.0, float(len(st.session_state.preprocessed_data['data'][0]) / 
                                                                    st.session_state.preprocessed_data['sfreq'])))
                    
                    seizure_times.append((start_time, end_time))
            else:
                seizure_times = None
            
            submit_button = st.form_submit_button("Extract Features")
            
            if submit_button:
                with st.spinner('Extracting features...'):
                    try:
                        # Get feature extraction parameters
                        feature_params = {
                            'time_domain': {
                                'mean': extract_mean,
                                'variance': extract_variance,
                                'skewness': extract_skewness,
                                'kurtosis': extract_kurtosis,
                                'line_length': extract_line_length,
                                'hjorth': extract_hjorth
                            },
                            'frequency_domain': {
                                'bands': extract_bands,
                                'spectral_entropy': extract_spectral_entropy,
                                'psd': extract_psd,
                                'coherence': extract_coherence
                            },
                            'time_frequency_domain': {
                                'wavelet': extract_wavelet,
                                'hilbert': extract_hilbert
                            },
                            'window_size': window_size,
                            'window_overlap': window_overlap / 100.0,
                            'seizure_times': seizure_times
                        }
                        
                        # Extract features
                        features, labels, feature_names = extract_features(
                            st.session_state.preprocessed_data['data'],
                            st.session_state.preprocessed_data['sfreq'],
                            st.session_state.preprocessed_data['ch_names'],
                            feature_params
                        )
                        
                        if features is not None:
                            st.session_state.features = features
                            st.session_state.labels = labels
                            st.session_state.feature_names = feature_names
                            
                            st.success(f"Successfully extracted {features.shape[1]} features from {features.shape[0]} windows")
                            
                            # Display feature visualization
                            st.subheader("Feature Visualization")
                            fig = plot_features(features, labels, feature_names)
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during feature extraction: {str(e)}")
        
        # Display current features if they exist
        if hasattr(st.session_state, 'features') and st.session_state.features is not None:
            st.subheader("Extracted Features Summary")
            
            # Display label distribution
            if st.session_state.labels is not None:
                label_counts = np.bincount(st.session_state.labels)
                
                # Check if there are two classes (seizure and non-seizure) or just one
                if len(label_counts) > 1:
                    st.write(f"Class distribution: {label_counts[0]} non-seizure windows, {label_counts[1]} seizure windows")
                else:
                    # Only one class found
                    class_type = "non-seizure" if 0 in st.session_state.labels else "seizure"
                    st.write(f"Class distribution: {label_counts[0]} {class_type} windows (only one class detected)")
            
            # Option to continue to model training
            if st.button("Next: Model Training →"):
                go_to_page("Model Training")

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if not hasattr(st.session_state, 'features') or st.session_state.features is None:
        st.warning("No features available. Please complete feature extraction first.")
        if st.button("← Back to Feature Extraction"):
            go_to_page("Feature Extraction")
    else:
        st.markdown("""
        Train a machine learning model to detect epilepsy patterns in the EEG data.
        """)
        
        with st.form("model_training_form"):
            # Model selection
            st.subheader("Model Selection")
            model_type = st.selectbox(
                "Select Machine Learning Model",
                ["Random Forest", "Support Vector Machine", "XGBoost", "Logistic Regression", "Neural Network"]
            )
            
            # Training parameters
            st.subheader("Training Parameters")
            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)
            cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1)
            
            # Class balancing
            class_balance = st.checkbox("Apply Class Balancing", True)
            balance_method = st.selectbox(
                "Class Balancing Method",
                ["SMOTE", "Random Undersampling", "Class Weights"],
                disabled=not class_balance
            )
            
            # Feature selection
            apply_feature_selection = st.checkbox("Apply Feature Selection", True)
            feature_selection_method = st.selectbox(
                "Feature Selection Method",
                ["Recursive Feature Elimination", "Select K Best", "Principal Component Analysis"],
                disabled=not apply_feature_selection
            )
            
            if apply_feature_selection:
                if feature_selection_method == "Select K Best" or feature_selection_method == "Recursive Feature Elimination":
                    n_features = st.slider(
                        "Number of Features to Select", 
                        min_value=5, 
                        max_value=min(50, st.session_state.features.shape[1]), 
                        value=min(20, st.session_state.features.shape[1]), 
                        step=5
                    )
                elif feature_selection_method == "Principal Component Analysis":
                    variance_retained = st.slider("Variance to Retain (%)", min_value=70, max_value=99, value=95, step=5)
            
            # Hyperparameter tuning
            apply_hyperparameter_tuning = st.checkbox("Apply Hyperparameter Tuning", True)
            
            submit_button = st.form_submit_button("Train Model")
            
            if submit_button:
                with st.spinner('Training model...'):
                    try:
                        # Get training parameters
                        train_params = {
                            'model_type': model_type,
                            'test_size': test_size / 100.0,
                            'cv_folds': cv_folds,
                            'class_balance': {
                                'apply': class_balance,
                                'method': balance_method if class_balance else None
                            },
                            'feature_selection': {
                                'apply': apply_feature_selection,
                                'method': feature_selection_method if apply_feature_selection else None,
                                'n_features': n_features if apply_feature_selection and 
                                              (feature_selection_method == "Select K Best" or 
                                               feature_selection_method == "Recursive Feature Elimination") else None,
                                'variance_retained': variance_retained / 100.0 if apply_feature_selection and 
                                                    feature_selection_method == "Principal Component Analysis" else None
                            },
                            'hyperparameter_tuning': apply_hyperparameter_tuning
                        }
                        
                        # Train model
                        model, predictions, metrics, feature_importances = train_model(
                            st.session_state.features,
                            st.session_state.labels,
                            st.session_state.feature_names if hasattr(st.session_state, 'feature_names') else None,
                            train_params
                        )
                        
                        if model is not None:
                            st.session_state.model = model
                            st.session_state.predictions = predictions
                            st.session_state.metrics = metrics
                            st.session_state.feature_importances = feature_importances
                            
                            st.success("Model training completed successfully")
                            
                            # Display metrics
                            st.subheader("Model Performance Metrics")
                            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                            st.write(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
                            st.write(f"Specificity: {metrics['specificity']:.4f}")
                            st.write(f"F1 Score: {metrics['f1_score']:.4f}")
                            
                            # Display confusion matrix
                            st.subheader("Confusion Matrix")
                            fig = plot_confusion_matrix(metrics['confusion_matrix'])
                            st.pyplot(fig)
                            
                            # Display ROC curve
                            st.subheader("ROC Curve")
                            fig = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
                            st.pyplot(fig)
                            
                            # Display feature importances if available
                            if feature_importances is not None:
                                st.subheader("Feature Importances")
                                importances_df = pd.DataFrame({
                                    'Feature': feature_importances['names'],
                                    'Importance': feature_importances['values']
                                }).sort_values('Importance', ascending=False)
                                
                                st.bar_chart(importances_df.set_index('Feature'))
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
        
        # Display current model results if they exist
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            st.subheader("Current Model Summary")
            
            st.write(f"Model type: {type(st.session_state.model).__name__}")
            
            # Option to continue to results
            if st.button("Next: View Results →"):
                go_to_page("Results")

# Results page
elif page == "Results":
    st.header("Results and Interpretation")
    
    if not hasattr(st.session_state, 'model') or st.session_state.model is None:
        st.warning("No model results available. Please complete model training first.")
        if st.button("← Back to Model Training"):
            go_to_page("Model Training")
    else:
        # Display comprehensive results
        st.subheader("Model Performance Summary")
        
        # Performance metrics
        metrics = st.session_state.metrics
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Sensitivity", f"{metrics['sensitivity']:.2%}")
        with col3:
            st.metric("Specificity", f"{metrics['specificity']:.2%}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            
        # Additional metrics
        st.subheader("Additional Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Precision: {metrics['precision']:.4f}")
            st.write(f"ROC AUC: {metrics['roc_auc']:.4f}")
            st.write(f"Average Precision: {metrics['average_precision']:.4f}")
        
        with col2:
            st.write(f"Positive Predictive Value: {metrics['ppv']:.4f}")
            st.write(f"Negative Predictive Value: {metrics['npv']:.4f}")
            st.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        # Display confusion matrix and ROC curve side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(metrics['confusion_matrix'])
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC Curve")
            fig = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
            st.pyplot(fig)
        
        # Brain activity visualization
        st.subheader("Brain Activity Patterns")
        if hasattr(st.session_state, 'preprocessed_data') and st.session_state.preprocessed_data is not None:
            # Create a brain activity visualization
            fig = plot_brain_activity(
                st.session_state.preprocessed_data['data'], 
                st.session_state.preprocessed_data['ch_names'],
                st.session_state.predictions if hasattr(st.session_state, 'predictions') else None
            )
            st.pyplot(fig)
        else:
            st.warning("Preprocessed data not available for brain activity visualization")
        
        # Model interpretation
        st.subheader("Model Interpretation")
        
        # Feature importances
        if hasattr(st.session_state, 'feature_importances') and st.session_state.feature_importances is not None:
            importances_df = pd.DataFrame({
                'Feature': st.session_state.feature_importances['names'],
                'Importance': st.session_state.feature_importances['values']
            }).sort_values('Importance', ascending=False)
            
            # Display top 10 features
            st.write("Top 10 Most Important Features:")
            st.bar_chart(importances_df.head(10).set_index('Feature'))
            
            # Group features by type
            if hasattr(st.session_state, 'feature_names') and st.session_state.feature_names is not None:
                st.write("Feature Importance by Category:")
                
                # Group features by their prefix (assuming format like "time_mean_channel1")
                feature_categories = {}
                for feature, importance in zip(st.session_state.feature_importances['names'], 
                                              st.session_state.feature_importances['values']):
                    category = feature.split('_')[0] if '_' in feature else 'other'
                    if category not in feature_categories:
                        feature_categories[category] = 0
                    feature_categories[category] += importance
                
                # Create category importance dataframe
                category_df = pd.DataFrame({
                    'Category': list(feature_categories.keys()),
                    'Importance': list(feature_categories.values())
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(category_df.set_index('Category'))
        
        # Clinical interpretation
        st.subheader("Clinical Interpretation")
        st.markdown("""
        ### Key Findings
        
        Based on the model performance and feature analysis, the following patterns appear significant for epilepsy detection:
        
        1. **Frequency Band Power**: Abnormal power distribution across frequency bands, particularly in the theta (4-8 Hz) and delta (1-4 Hz) ranges, which are often elevated during epileptic activity.
        
        2. **Signal Variance**: Higher signal variance in certain channels may indicate the presence of epileptiform discharges.
        
        3. **Wavelet Features**: Time-frequency characteristics captured by wavelet transforms highlight transient events characteristic of epileptic seizures.
        
        4. **Brain Region Activity**: Specific channels/regions show distinctive patterns during seizure activity.
        
        ### Limitations
        
        - The model's performance depends on the quality and quantity of the input data
        - Individual variations in EEG patterns may affect accuracy
        - The model should be used as a supporting tool for clinical diagnosis, not as a replacement for medical expertise
        """)
        
        # Download options
        st.subheader("Download Results")
        
        # Create a temporary results dictionary
        results = {
            'metrics': metrics,
            'model_type': type(st.session_state.model).__name__,
            'feature_importances': st.session_state.feature_importances if hasattr(st.session_state, 'feature_importances') else None
        }
        
        # Convert to JSON
        results_json = pd.Series(results).to_json()
        
        # Download button
        st.download_button(
            label="Download Results as JSON",
            data=results_json,
            file_name="epilepsy_detection_results.json",
            mime="application/json"
        )
