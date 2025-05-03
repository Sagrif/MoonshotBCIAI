import numpy as np
import mne
import pandas as pd
import os
from scipy import signal
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_eeg_data(file_path):
    """
    Load EEG data from various file formats
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file
    
    Returns:
    --------
    data : ndarray
        EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load data based on file extension
        if file_ext in ['.edf', '.bdf']:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
        
        elif file_ext == '.vhdr':
            raw = mne.io.read_raw_brainvision(file_path, preload=True)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
        
        elif file_ext == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
        
        elif file_ext == '.cnt':
            raw = mne.io.read_raw_cnt(file_path, preload=True)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.ch_names
        
        elif file_ext in ['.txt', '.csv']:
            # Assume a simple format where each column is a channel
            # First row could be channel names
            df = pd.read_csv(file_path)
            
            # Check if the first column might be time
            first_col = df.columns[0].lower()
            if first_col in ['time', 't', 'timestamp']:
                # Extract sampling frequency from time differences
                time_diffs = np.diff(df.iloc[:, 0].values)
                sfreq = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 250.0
                
                # Use the remaining columns as EEG data
                data = df.iloc[:, 1:].values.T
                ch_names = df.columns[1:].tolist()
            else:
                # Assume a constant sampling rate of 250 Hz if not specified
                sfreq = 250.0
                data = df.values.T
                ch_names = df.columns.tolist()
        
        else:
            logging.error(f"Unsupported file format: {file_ext}")
            return None, None, None
        
        logging.info(f"Successfully loaded data: {data.shape[0]} channels, {data.shape[1]} samples at {sfreq} Hz")
        return data, sfreq, ch_names
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None

def preprocess_data(data, sfreq, ch_names, params):
    """
    Preprocess EEG data
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    params : dict
        Preprocessing parameters
    
    Returns:
    --------
    preprocessed_data : dict
        Dictionary containing the preprocessed data, sfreq, and ch_names
    """
    try:
        # Create a copy of the data
        processed_data = data.copy()
        processed_sfreq = sfreq
        processed_ch_names = ch_names.copy()
        
        # Create MNE Raw object for better preprocessing
        info = mne.create_info(ch_names=processed_ch_names, sfreq=processed_sfreq, ch_types='eeg')
        raw = mne.io.RawArray(processed_data, info)
        
        # Apply notch filter to remove power line noise
        if params['apply_notch']:
            notch_freq = params['notch_freq']
            raw.notch_filter(freqs=notch_freq, picks='all')
            logging.info(f"Applied notch filter at {notch_freq} Hz")
        
        # Apply bandpass filter
        if params['apply_bandpass']:
            l_freq = params['bandpass_low']
            h_freq = params['bandpass_high']
            raw.filter(l_freq=l_freq, h_freq=h_freq, picks='all')
            logging.info(f"Applied bandpass filter from {l_freq} Hz to {h_freq} Hz")
        
        # Re-reference the data
        if params['apply_reference']:
            if params['reference_type'] == 'Average':
                raw.set_eeg_reference(ref_channels='average')
                logging.info("Applied average reference")
            elif params['reference_type'] == 'Mastoids':
                # Look for mastoid channels
                mastoid_chs = [ch for ch in processed_ch_names if any(m in ch.lower() for m in ['m1', 'm2', 'a1', 'a2', 'mastoid'])]
                if mastoid_chs:
                    raw.set_eeg_reference(ref_channels=mastoid_chs)
                    logging.info(f"Applied mastoid reference using channels: {mastoid_chs}")
                else:
                    logging.warning("No mastoid channels found. Skipping re-referencing.")
        
        # Remove artifacts
        if params['remove_artifacts']:
            if params['artifact_method'] == 'ICA':
                # Apply ICA for artifact removal
                ica = mne.preprocessing.ICA(n_components=min(15, len(processed_ch_names) - 1), random_state=42)
                ica.fit(raw)
                
                # Find components that correlate with EOG/EMG
                # This is a simplified method - in practice, would need more sophisticated detection
                exclude = []
                for idx, component in enumerate(ica.get_components()[:5]):
                    # Simple heuristic: components with very high variance might be artifacts
                    if np.var(component) > 3 * np.mean(np.var(ica.get_components(), axis=1)):
                        exclude.append(idx)
                
                ica.exclude = exclude
                ica.apply(raw)
                logging.info(f"Applied ICA artifact removal. Excluded components: {exclude}")
            
            elif params['artifact_method'] == 'Threshold':
                # Simple threshold-based artifact removal
                # Mark segments where the signal exceeds 100 μV
                annotations = mne.Annotations([], [], [])
                data = raw.get_data()
                
                for ch_idx, ch_name in enumerate(processed_ch_names):
                    # Find segments where signal exceeds threshold
                    threshold = 100e-6  # 100 μV
                    bad_segments = np.where(np.abs(data[ch_idx]) > threshold)[0]
                    
                    if len(bad_segments) > 0:
                        # Group consecutive points into segments
                        diff = np.diff(bad_segments)
                        segment_indices = np.where(diff > 1)[0] + 1
                        segment_start = np.concatenate([[0], segment_indices])
                        segment_end = np.concatenate([segment_indices, [len(bad_segments)]])
                        
                        for start, end in zip(segment_start, segment_end):
                            onset = bad_segments[start] / processed_sfreq
                            duration = (bad_segments[end-1] - bad_segments[start]) / processed_sfreq + 0.1
                            annotations.append(onset, duration, 'bad')
                
                raw.set_annotations(annotations)
                logging.info(f"Applied threshold-based artifact marking. {len(annotations)} segments marked.")
        
        # Resample the data if requested
        if params['resample'] and params['resample_freq'] is not None:
            raw.resample(params['resample_freq'])
            processed_sfreq = params['resample_freq']
            logging.info(f"Resampled data to {processed_sfreq} Hz")
        
        # Get the processed data
        processed_data = raw.get_data()
        
        # Return preprocessed data
        preprocessed_data = {
            'data': processed_data,
            'sfreq': processed_sfreq,
            'ch_names': processed_ch_names
        }
        
        return preprocessed_data
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        return None
