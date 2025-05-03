import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
import mne
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(data, sfreq, ch_names, params):
    """
    Extract features from EEG data
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    params : dict
        Feature extraction parameters
    
    Returns:
    --------
    features : ndarray
        Extracted features with shape (n_windows, n_features)
    labels : ndarray
        Labels for each window (0 for non-seizure, 1 for seizure)
    feature_names : list
        Names of the extracted features
    """
    try:
        # Get parameters
        window_size = params['window_size']  # in seconds
        window_overlap = params['window_overlap']  # as a fraction
        seizure_times = params['seizure_times']
        
        # Calculate window parameters
        n_samples = data.shape[1]
        window_samples = int(window_size * sfreq)
        step_samples = int(window_samples * (1 - window_overlap))
        n_windows = int(np.floor((n_samples - window_samples) / step_samples) + 1)
        
        logging.info(f"Extracting features from {n_windows} windows")
        
        # Initialize features list and feature names
        all_features = []
        all_feature_names = []
        
        # Loop through each window
        for win_idx in range(n_windows):
            start_sample = win_idx * step_samples
            end_sample = start_sample + window_samples
            
            if end_sample > n_samples:
                break
            
            # Extract window data
            window_data = data[:, start_sample:end_sample]
            
            # Initialize features for this window
            window_features = []
            
            # Extract time domain features
            if any(params['time_domain'].values()):
                time_features, time_feature_names = extract_time_domain_features(
                    window_data, ch_names, params['time_domain']
                )
                window_features.extend(time_features)
                
                # Add feature names only for the first window
                if win_idx == 0:
                    all_feature_names.extend(time_feature_names)
            
            # Extract frequency domain features
            if any(params['frequency_domain'].values()):
                freq_features, freq_feature_names = extract_frequency_domain_features(
                    window_data, sfreq, ch_names, params['frequency_domain']
                )
                window_features.extend(freq_features)
                
                # Add feature names only for the first window
                if win_idx == 0:
                    all_feature_names.extend(freq_feature_names)
            
            # Extract time-frequency domain features
            if any(params['time_frequency_domain'].values()):
                tf_features, tf_feature_names = extract_time_frequency_features(
                    window_data, sfreq, ch_names, params['time_frequency_domain']
                )
                window_features.extend(tf_features)
                
                # Add feature names only for the first window
                if win_idx == 0:
                    all_feature_names.extend(tf_feature_names)
            
            # Add the window features to all features
            all_features.append(window_features)
        
        # Convert to numpy array
        features = np.array(all_features)
        
        # Generate labels based on seizure times
        labels = np.zeros(n_windows, dtype=int)
        
        if seizure_times is not None:
            for start_time, end_time in seizure_times:
                # Convert times to window indices
                start_window = int(np.floor(start_time / (window_size * (1 - window_overlap))))
                end_window = int(np.ceil(end_time / (window_size * (1 - window_overlap))))
                
                # Ensure indices are within bounds
                start_window = max(0, start_window)
                end_window = min(n_windows, end_window)
                
                # Mark windows within the seizure period
                labels[start_window:end_window] = 1
        
        # Scale features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        logging.info(f"Extracted {features.shape[1]} features from {features.shape[0]} windows")
        logging.info(f"Label distribution: {np.bincount(labels)}")
        
        return features, labels, all_feature_names
    
    except Exception as e:
        logging.error(f"Error during feature extraction: {str(e)}")
        return None, None, None

def extract_time_domain_features(window_data, ch_names, time_params):
    """
    Extract time domain features from a window of EEG data
    
    Parameters:
    -----------
    window_data : ndarray
        Window of EEG data with shape (n_channels, n_samples)
    ch_names : list
        Channel names
    time_params : dict
        Time domain feature parameters
    
    Returns:
    --------
    features : list
        Extracted time domain features
    feature_names : list
        Names of the extracted features
    """
    features = []
    feature_names = []
    
    # Loop through channels
    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = window_data[ch_idx]
        
        # Mean
        if time_params['mean']:
            mean = np.mean(ch_data)
            features.append(mean)
            feature_names.append(f"time_mean_{ch_name}")
        
        # Variance
        if time_params['variance']:
            variance = np.var(ch_data)
            features.append(variance)
            feature_names.append(f"time_variance_{ch_name}")
        
        # Skewness
        if time_params['skewness']:
            skewness = stats.skew(ch_data)
            features.append(skewness)
            feature_names.append(f"time_skewness_{ch_name}")
        
        # Kurtosis
        if time_params['kurtosis']:
            kurtosis = stats.kurtosis(ch_data)
            features.append(kurtosis)
            feature_names.append(f"time_kurtosis_{ch_name}")
        
        # Line length
        if time_params['line_length']:
            line_length = np.sum(np.abs(np.diff(ch_data)))
            features.append(line_length)
            feature_names.append(f"time_line_length_{ch_name}")
        
        # Hjorth parameters
        if time_params['hjorth']:
            # Activity (variance of the signal)
            activity = np.var(ch_data)
            
            # Mobility (standard deviation of the first derivative divided by the standard deviation of the signal)
            first_deriv = np.diff(ch_data)
            mobility = np.std(first_deriv) / np.std(ch_data) if np.std(ch_data) > 0 else 0
            
            # Complexity (mobility of the first derivative divided by the mobility of the signal)
            second_deriv = np.diff(first_deriv)
            mob_first_deriv = np.std(second_deriv) / np.std(first_deriv) if np.std(first_deriv) > 0 else 0
            complexity = mob_first_deriv / mobility if mobility > 0 else 0
            
            features.extend([mobility, complexity])
            feature_names.extend([f"time_hjorth_mobility_{ch_name}", f"time_hjorth_complexity_{ch_name}"])
    
    return features, feature_names

def extract_frequency_domain_features(window_data, sfreq, ch_names, freq_params):
    """
    Extract frequency domain features from a window of EEG data
    
    Parameters:
    -----------
    window_data : ndarray
        Window of EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    freq_params : dict
        Frequency domain feature parameters
    
    Returns:
    --------
    features : list
        Extracted frequency domain features
    feature_names : list
        Names of the extracted features
    """
    features = []
    feature_names = []
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    
    # Loop through channels
    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = window_data[ch_idx]
        
        # Calculate power spectrum
        freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(256, len(ch_data)))
        
        # Frequency bands power
        if freq_params['bands']:
            for band_name, (fmin, fmax) in bands.items():
                # Find frequencies within the band
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                
                # Calculate band power
                band_power = np.sum(psd[idx_band])
                
                features.append(band_power)
                feature_names.append(f"freq_band_power_{band_name}_{ch_name}")
                
                # Calculate relative band power
                total_power = np.sum(psd)
                rel_band_power = band_power / total_power if total_power > 0 else 0
                
                features.append(rel_band_power)
                feature_names.append(f"freq_rel_band_power_{band_name}_{ch_name}")
        
        # Spectral entropy
        if freq_params['spectral_entropy']:
            # Normalize PSD to get probability distribution
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.ones_like(psd) / len(psd)
            
            # Calculate spectral entropy
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            features.append(spectral_entropy)
            feature_names.append(f"freq_spectral_entropy_{ch_name}")
        
        # Power spectral density features
        if freq_params['psd']:
            # Spectral edge frequency (below which 95% of the power resides)
            cumulative_power = np.cumsum(psd) / np.sum(psd) if np.sum(psd) > 0 else np.arange(len(psd)) / len(psd)
            idx_edge = np.argmax(cumulative_power >= 0.95)
            spectral_edge = freqs[idx_edge] if idx_edge < len(freqs) else freqs[-1]
            
            features.append(spectral_edge)
            feature_names.append(f"freq_spectral_edge_{ch_name}")
            
            # Spectral peak frequency (frequency with maximum power)
            peak_freq = freqs[np.argmax(psd)]
            
            features.append(peak_freq)
            feature_names.append(f"freq_peak_freq_{ch_name}")
    
    # Coherence between channels
    if freq_params['coherence'] and len(ch_names) >= 2:
        # Calculate coherence for pairs of channels
        for i in range(len(ch_names)):
            for j in range(i+1, len(ch_names)):
                ch_i_data = window_data[i]
                ch_j_data = window_data[j]
                
                # Calculate cross-spectral density
                freqs, csd = signal.csd(ch_i_data, ch_j_data, sfreq, nperseg=min(256, len(ch_i_data)))
                
                # Calculate PSDs for both channels
                _, psd_i = signal.welch(ch_i_data, sfreq, nperseg=min(256, len(ch_i_data)))
                _, psd_j = signal.welch(ch_j_data, sfreq, nperseg=min(256, len(ch_j_data)))
                
                # Calculate coherence
                coherence = np.abs(csd)**2 / (psd_i * psd_j)
                
                # Average coherence in each frequency band
                for band_name, (fmin, fmax) in bands.items():
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_coherence = np.mean(coherence[idx_band])
                    
                    features.append(band_coherence)
                    feature_names.append(f"freq_coherence_{band_name}_{ch_names[i]}_{ch_names[j]}")
    
    return features, feature_names

def extract_time_frequency_features(window_data, sfreq, ch_names, tf_params):
    """
    Extract time-frequency domain features from a window of EEG data
    
    Parameters:
    -----------
    window_data : ndarray
        Window of EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    tf_params : dict
        Time-frequency domain feature parameters
    
    Returns:
    --------
    features : list
        Extracted time-frequency domain features
    feature_names : list
        Names of the extracted features
    """
    features = []
    feature_names = []
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    
    # Loop through channels
    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = window_data[ch_idx]
        
        # Wavelet coefficients
        if tf_params['wavelet']:
            # Choose wavelet type
            wavelet = 'db4'
            
            # Get maximum decomposition level
            max_level = pywt.dwt_max_level(len(ch_data), wavelet)
            level = min(5, max_level)  # Use up to 5 levels
            
            # Perform discrete wavelet transform
            coeffs = pywt.wavedec(ch_data, wavelet, level=level)
            
            # Extract features from each level
            for i, coef in enumerate(coeffs):
                # Absolute mean
                abs_mean = np.mean(np.abs(coef))
                features.append(abs_mean)
                feature_names.append(f"tf_wavelet_abs_mean_level{i}_{ch_name}")
                
                # Energy
                energy = np.sum(coef**2)
                features.append(energy)
                feature_names.append(f"tf_wavelet_energy_level{i}_{ch_name}")
                
                # Standard deviation
                std = np.std(coef)
                features.append(std)
                feature_names.append(f"tf_wavelet_std_level{i}_{ch_name}")
        
        # Hilbert transform
        if tf_params['hilbert']:
            # Calculate analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(ch_data)
            
            # Calculate envelope and instantaneous phase
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sfreq
            
            # Extract features from envelope
            env_mean = np.mean(amplitude_envelope)
            env_std = np.std(amplitude_envelope)
            env_max = np.max(amplitude_envelope)
            
            features.extend([env_mean, env_std, env_max])
            feature_names.extend([
                f"tf_hilbert_env_mean_{ch_name}",
                f"tf_hilbert_env_std_{ch_name}",
                f"tf_hilbert_env_max_{ch_name}"
            ])
            
            # Extract features from instantaneous frequency
            if len(instantaneous_frequency) > 0:
                freq_mean = np.mean(instantaneous_frequency)
                freq_std = np.std(instantaneous_frequency)
                freq_max = np.max(instantaneous_frequency)
                
                features.extend([freq_mean, freq_std, freq_max])
                feature_names.extend([
                    f"tf_hilbert_freq_mean_{ch_name}",
                    f"tf_hilbert_freq_std_{ch_name}",
                    f"tf_hilbert_freq_max_{ch_name}"
                ])
    
    return features, feature_names
