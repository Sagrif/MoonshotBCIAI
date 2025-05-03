import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_example_description():
    """
    Create example description for the application with visualizations
    """
    st.markdown("""
    ## Example EEG Patterns
    
    Below are examples of typical EEG patterns seen in epilepsy. These visualizations help you understand what the model is looking for.
    """)
    
    # Create tabs for different pattern visualizations
    tab1, tab2, tab3 = st.tabs(["Normal EEG", "Epileptiform Discharges", "Seizure Activity"])
    
    with tab1:
        st.markdown("""
        ### Normal EEG
        
        Normal EEG is characterized by:
        - Organized background rhythm
        - No sharp waves or spikes
        - Alpha activity (8-13 Hz) during relaxed wakefulness
        - Beta activity (13-30 Hz) during active thinking
        - No abnormal slow waves in wakefulness
        """)
        
        # Create a visualization of normal EEG
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Simulate normal EEG (alpha rhythm with some beta)
        t = np.linspace(0, 4, 1000)
        
        # Alpha rhythm (8-13 Hz)
        alpha = np.sin(2 * np.pi * 10 * t) * 0.5
        
        # Beta components (13-30 Hz)
        beta = np.sin(2 * np.pi * 20 * t) * 0.2
        
        # Add some random noise
        noise = np.random.normal(0, 0.1, len(t))
        
        # Combine signals
        eeg = alpha + beta + noise
        
        # Plot
        ax.plot(t, eeg, 'k-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Normal EEG with Alpha Rhythm')
        ax.grid(True)
        
        st.pyplot(fig)
    
    with tab2:
        st.markdown("""
        ### Epileptiform Discharges
        
        Epileptiform discharges include:
        - Spikes: Pointed transients with duration of 20-70 ms
        - Sharp waves: Pointed transients with duration of 70-200 ms
        - Spike-and-wave complexes: Spike followed by a slow wave
        - These are not seizures but indicate epileptic tendency
        """)
        
        # Create a visualization of epileptiform discharges
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Simulate background EEG
        t = np.linspace(0, 4, 1000)
        background = np.sin(2 * np.pi * 10 * t) * 0.3 + np.random.normal(0, 0.1, len(t))
        
        # Add epileptiform discharges (spikes)
        eeg = background.copy()
        
        # Function to create a spike
        def create_spike(t_center, width, amplitude):
            return amplitude * np.exp(-((t - t_center) ** 2) / (2 * width ** 2))
        
        # Add several spikes
        spike_times = [0.5, 1.2, 2.3, 3.1]
        for spike_t in spike_times:
            # Add a spike
            eeg += create_spike(spike_t, 0.01, 1.0)
            # Add a slow wave after the spike
            eeg += create_spike(spike_t + 0.08, 0.04, -0.5)
        
        # Plot
        ax.plot(t, eeg, 'k-', linewidth=0.8)
        
        # Mark spikes
        for spike_t in spike_times:
            ax.annotate('Spike', xy=(spike_t, 1.0), xytext=(spike_t, 1.5),
                       arrowprops=dict(facecolor='red', shrink=0.05))
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('EEG with Epileptiform Discharges (Spikes)')
        ax.grid(True)
        
        st.pyplot(fig)
    
    with tab3:
        st.markdown("""
        ### Seizure Activity
        
        Seizure patterns in EEG show:
        - Abnormal, hypersynchronous electrical activity
        - Evolution of frequency and amplitude over time
        - Various patterns such as:
          - Rhythmic spike-and-wave discharges
          - Low-voltage fast activity
          - Rhythmic slowing with increasing amplitude
        """)
        
        # Create a visualization of seizure activity
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Simulate seizure EEG
        t = np.linspace(0, 10, 2500)
        
        # Create pre-ictal, ictal, and post-ictal phases
        pre_ictal = np.sin(2 * np.pi * 10 * t[:500]) * 0.3 + np.random.normal(0, 0.1, 500)
        
        # Seizure onset - transition to faster activity with increasing amplitude
        seizure_onset = np.zeros(500)
        for i in range(500):
            # Gradually increase frequency and amplitude
            freq = 3 + i * 0.05
            amp = 0.3 + i * 0.004
            seizure_onset[i] = amp * np.sin(2 * np.pi * freq * t[500+i]) + np.random.normal(0, 0.05, 1)
        
        # Ictal (seizure) - high amplitude, rhythmic activity
        ictal = np.zeros(1000)
        for i in range(1000):
            # Rhythmic spike-and-wave pattern
            cycle = i % 20
            if cycle < 3:
                # Spike
                ictal[i] = 1.5 + np.random.normal(0, 0.1, 1)
            else:
                # Wave
                ictal[i] = -0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * cycle) + np.random.normal(0, 0.1, 1)
        
        # Post-ictal - suppression and slow recovery
        post_ictal = np.linspace(0, 0.3, 500) * np.sin(2 * np.pi * 3 * t[-500:]) + np.random.normal(0, 0.1, 500)
        
        # Combine phases
        eeg = np.concatenate([pre_ictal, seizure_onset, ictal, post_ictal])
        
        # Plot
        ax.plot(t, eeg, 'k-', linewidth=0.8)
        
        # Mark different phases
        ax.axvspan(0, 2, alpha=0.2, color='green', label='Pre-ictal')
        ax.axvspan(2, 4, alpha=0.2, color='yellow', label='Seizure Onset')
        ax.axvspan(4, 8, alpha=0.2, color='red', label='Ictal (Seizure)')
        ax.axvspan(8, 10, alpha=0.2, color='blue', label='Post-ictal')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('EEG during Seizure Activity')
        ax.grid(True)
        ax.legend(loc='upper right')
        
        st.pyplot(fig)
    
    st.markdown("""
    ## Features Used for Epilepsy Detection
    
    The model extracts and analyzes multiple features from EEG data:
    
    ### Time Domain Features
    - **Mean and Variance**: Capture amplitude characteristics
    - **Skewness and Kurtosis**: Measure distribution asymmetry and peakedness
    - **Line Length**: Quantifies signal complexity and is sensitive to spikes
    - **Hjorth Parameters**: Measure signal complexity and smoothness
    
    ### Frequency Domain Features
    - **Band Powers**: Energy in delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30+ Hz) bands
    - **Spectral Entropy**: Measures irregularity in the frequency domain
    - **Peak Frequency**: Dominant frequency component
    - **Spectral Edge Frequency**: Frequency below which most signal power is contained
    
    ### Time-Frequency Features
    - **Wavelet Coefficients**: Capture time-frequency characteristics
    - **Hilbert Transform**: Analyzes instantaneous amplitude and frequency
    """)
    
    # Create example feature importance visualization
    st.subheader("Example Feature Importance in Epilepsy Detection")
    
    # Create mock feature importance data
    features = [
        'Theta Band Power', 'Line Length', 'Wavelet Energy',
        'Delta Band Power', 'Kurtosis', 'Alpha/Theta Ratio',
        'Hjorth Mobility', 'Spectral Entropy', 'Beta Band Power',
        'Signal Variance'
    ]
    
    importance = [0.23, 0.18, 0.15, 0.12, 0.09, 0.07, 0.06, 0.05, 0.03, 0.02]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Plot with Streamlit
    st.bar_chart(df.set_index('Feature'))
    
    st.markdown("""
    > Note: The above chart shows example feature importances. Actual importances will vary based on your specific data and model.
    """)

def create_demo_data(n_samples=1000, sfreq=250, n_channels=19, include_seizure=True):
    """
    Create demo EEG data for testing purposes
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    sfreq : float
        Sampling frequency
    n_channels : int
        Number of channels
    include_seizure : bool
        Whether to include simulated seizure patterns
    
    Returns:
    --------
    data : ndarray
        Demo EEG data with shape (n_channels, n_samples)
    ch_names : list
        Channel names
    """
    # Create time vector
    t = np.arange(n_samples) / sfreq
    
    # Create standard 10-20 channel names
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
        'Pz', 'P4', 'T6', 'O1', 'O2'
    ]
    
    # If requested channels are more than standard, add additional channel names
    if n_channels > len(ch_names):
        additional = [f'Ch{i+1}' for i in range(len(ch_names), n_channels)]
        ch_names = ch_names + additional
    else:
        # Limit to requested number of channels
        ch_names = ch_names[:n_channels]
    
    # Initialize data
    data = np.zeros((n_channels, n_samples))
    
    # Generate alpha rhythm (8-13 Hz) for all channels
    alpha = np.sin(2 * np.pi * 10 * t) * 0.5
    
    # Generate beta components (13-30 Hz) for frontal channels
    beta = np.sin(2 * np.pi * 20 * t) * 0.2
    
    # Generate theta components (4-8 Hz) for all channels
    theta = np.sin(2 * np.pi * 6 * t) * 0.3
    
    # Generate delta components (1-4 Hz) for all channels
    delta = np.sin(2 * np.pi * 2 * t) * 0.4
    
    # Add components to channels with appropriate weighting
    for i, ch in enumerate(ch_names):
        # Base activity for all channels
        data[i] = alpha + theta + delta
        
        # Add beta for frontal channels
        if ch in ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']:
            data[i] += beta * 1.5
        
        # Add more alpha for occipital channels
        if ch in ['O1', 'O2']:
            data[i] += alpha * 2
        
        # Add more theta for temporal channels
        if ch in ['T3', 'T4', 'T5', 'T6']:
            data[i] += theta * 1.5
        
        # Add random noise (different for each channel)
        data[i] += np.random.normal(0, 0.1, n_samples)
    
    # Add seizure-like activity if requested
    if include_seizure:
        # Create seizure time windows (start in the middle third of the recording)
        seizure_start = n_samples // 3
        seizure_duration = n_samples // 6  # 1/6 of the total duration
        seizure_end = seizure_start + seizure_duration
        
        # Function to create a spike
        def create_spike(t_center, width, amplitude):
            return amplitude * np.exp(-((t - t_center) ** 2) / (2 * width ** 2))
        
        # Create 3 Hz spike-and-wave pattern (classic epileptiform pattern)
        seizure = np.zeros(n_samples)
        
        # Generate spike-and-wave discharges at 3 Hz
        for j in range(seizure_start, seizure_end, int(sfreq/3)):  # 3 Hz rhythm
            # Add a spike
            seizure += create_spike(t[j], 0.015, 2.0)
            # Add a slow wave after the spike
            seizure += create_spike(t[min(j + int(sfreq/10), n_samples-1)], 0.05, -1.0)
        
        # Add higher frequency oscillations overlaid
        fast_oscillation = np.zeros(n_samples)
        fast_oscillation[seizure_start:seizure_end] = np.sin(2 * np.pi * 20 * t[seizure_start:seizure_end]) * 0.5
        
        # Add increasing amplitude trend in the middle of the seizure
        amplitude_mod = np.zeros(n_samples)
        mid_seizure_start = seizure_start + seizure_duration//4
        mid_seizure_end = seizure_end - seizure_duration//4
        amplitude_mod[mid_seizure_start:mid_seizure_end] = np.linspace(0, 1.5, mid_seizure_end-mid_seizure_start)
        
        # Apply seizure patterns to channels with appropriate weighting
        for i, ch in enumerate(ch_names):
            # Temporal and central channels typically show more seizure activity
            if ch in ['T3', 'T4', 'C3', 'C4', 'Cz', 'T5', 'T6']:
                # Higher amplitude in these channels
                data[i] += seizure * (1.0 + 0.5 * np.random.random()) + fast_oscillation + amplitude_mod
            else:
                # Reduced amplitude in other channels (representing spread)
                data[i] += seizure * (0.3 + 0.3 * np.random.random()) + fast_oscillation * 0.3 + amplitude_mod * 0.3
    
    return data, ch_names
