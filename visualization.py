import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_eeg_data(data, sfreq, ch_names, duration=10, n_channels=None, start_time=0):
    """
    Plot EEG data
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency
    ch_names : list
        Channel names
    duration : float
        Duration to plot in seconds
    n_channels : int
        Number of channels to plot
    start_time : float
        Start time in seconds
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    try:
        # Calculate samples to show
        start_sample = int(start_time * sfreq)
        n_samples = int(duration * sfreq)
        end_sample = min(start_sample + n_samples, data.shape[1])
        
        # Select channels to plot
        if n_channels is None:
            n_channels = data.shape[0]
        else:
            n_channels = min(n_channels, data.shape[0])
        
        # Create figure
        fig, ax = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.5), sharex=True)
        
        # Handle case with only one channel
        if n_channels == 1:
            ax = [ax]
        
        # Define time vector
        time = np.arange(start_sample, end_sample) / sfreq
        
        # Plot each channel
        for i in range(n_channels):
            channel_data = data[i, start_sample:end_sample]
            
            # Normalize for better visualization
            channel_data = channel_data - np.mean(channel_data)
            scaling = np.max(np.abs(channel_data)) if np.max(np.abs(channel_data)) > 0 else 1
            
            ax[i].plot(time, channel_data / scaling, 'k-', linewidth=0.5)
            ax[i].set_ylabel(ch_names[i])
            ax[i].grid(True)
            
            # Remove y-axis ticks for cleaner visualization
            ax[i].set_yticks([])
        
        # Add x-axis label
        ax[-1].set_xlabel('Time (s)')
        
        # Add title
        plt.suptitle(f'EEG Data (showing {n_channels}/{data.shape[0]} channels)', fontsize=16)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logging.error(f"Error plotting EEG data: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting EEG data: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig

def plot_features(features, labels, feature_names, max_features=20):
    """
    Plot feature visualization
    
    Parameters:
    -----------
    features : ndarray
        Feature matrix
    labels : ndarray
        Labels for each sample
    feature_names : list
        Names of the features
    max_features : int
        Maximum number of features to display in individual plots
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    try:
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        
        # Define layout
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Feature distributions
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Limit to top features
        if feature_names is not None and len(feature_names) > max_features:
            # Calculate feature importance using simple variance
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[-max_features:]
            
            # Select top features
            plot_features = features[:, top_indices]
            plot_feature_names = [feature_names[i] for i in top_indices]
        else:
            plot_features = features
            plot_feature_names = feature_names
        
        # Plot feature distributions as boxplots
        if labels is not None and len(np.unique(labels)) == 2:
            # Create dataframe for boxplot with classes
            data_to_plot = []
            positions = []
            colors = []
            
            for i in range(min(max_features, plot_features.shape[1])):
                # Class 0
                data_to_plot.append(plot_features[labels == 0, i])
                positions.append(i * 3)
                colors.append('blue')
                
                # Class 1
                data_to_plot.append(plot_features[labels == 1, i])
                positions.append(i * 3 + 1)
                colors.append('red')
            
            # Create boxplot
            bp = ax1.boxplot(data_to_plot, positions=positions, patch_artist=True, 
                           widths=0.8, showfliers=False)
            
            # Customize boxplot colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Set x-ticks in the middle of each feature's boxplots
            if plot_feature_names is not None:
                ax1.set_xticks([i * 3 + 0.5 for i in range(len(plot_feature_names))])
                ax1.set_xticklabels(plot_feature_names, rotation=90)
            
            # Add legend
            legend_elements = [
                Patch(facecolor='blue', alpha=0.6, label='Non-seizure'),
                Patch(facecolor='red', alpha=0.6, label='Seizure')
            ]
            ax1.legend(handles=legend_elements)
        else:
            # Simple boxplot without class distinction
            bp = ax1.boxplot(plot_features, patch_artist=True, widths=0.8, showfliers=False)
            
            # Customize boxplot color
            for patch in bp['boxes']:
                patch.set_facecolor('blue')
                patch.set_alpha(0.6)
            
            # Set x-ticks
            if plot_feature_names is not None:
                ax1.set_xticks(np.arange(1, len(plot_feature_names) + 1))
                ax1.set_xticklabels(plot_feature_names, rotation=90)
        
        ax1.set_title("Feature Distributions")
        ax1.grid(True, axis='y')
        
        # Feature correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate correlation matrix
        if plot_features.shape[1] > 1:
            corr_matrix = np.corrcoef(plot_features, rowvar=False)
            
            # Plot heatmap
            im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax2)
            
            # Set labels
            if plot_feature_names is not None:
                ax2.set_xticks(np.arange(len(plot_feature_names)))
                ax2.set_yticks(np.arange(len(plot_feature_names)))
                ax2.set_xticklabels(plot_feature_names, rotation=90)
                ax2.set_yticklabels(plot_feature_names)
            
            ax2.set_title("Feature Correlation Matrix")
        else:
            ax2.text(0.5, 0.5, "Correlation matrix requires at least 2 features", 
                   horizontalalignment='center', verticalalignment='center')
        
        # 2D dimensionality reduction (PCA or t-SNE)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if features.shape[1] >= 2 and features.shape[0] >= 3:
            # Apply PCA
            pca = PCA(n_components=2, random_state=42)
            pca_features = pca.fit_transform(features)
            
            # Plot PCA
            if labels is not None:
                scatter = ax3.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, 
                                   alpha=0.6, cmap='coolwarm', edgecolors='k', s=50)
                
                # Add legend
                legend1 = ax3.legend(*scatter.legend_elements(), title="Classes")
                ax3.add_artist(legend1)
            else:
                ax3.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.6, edgecolors='k', s=50)
            
            # Add variance explained
            var_exp = pca.explained_variance_ratio_
            ax3.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
            ax3.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
            ax3.set_title("PCA Dimensionality Reduction")
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "PCA requires at least 3 samples and 2 features", 
                   horizontalalignment='center', verticalalignment='center')
        
        # 3D dimensionality reduction
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        
        if features.shape[1] >= 3 and features.shape[0] >= 4:
            # Apply t-SNE
            try:
                # Try t-SNE but it might fail with too few samples
                if features.shape[0] >= 50:  # t-SNE works better with more samples
                    tsne = TSNE(n_components=3, random_state=42)
                    tsne_features = tsne.fit_transform(features)
                else:
                    # Use PCA for smaller datasets
                    pca = PCA(n_components=3, random_state=42)
                    tsne_features = pca.fit_transform(features)
                
                # Plot 3D scatter
                if labels is not None:
                    scatter = ax4.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2],
                                      c=labels, alpha=0.6, cmap='coolwarm', edgecolors='k', s=50)
                    
                    # Add legend
                    legend1 = ax4.legend(*scatter.legend_elements(), title="Classes")
                    ax4.add_artist(legend1)
                else:
                    ax4.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2],
                             alpha=0.6, edgecolors='k', s=50)
                
                # Set labels
                if isinstance(tsne, TSNE):
                    ax4.set_title("t-SNE 3D Visualization")
                else:
                    var_exp = pca.explained_variance_ratio_
                    ax4.set_title(f"PCA 3D (Total: {sum(var_exp):.1%} variance)")
                
                ax4.set_xlabel("Dimension 1")
                ax4.set_ylabel("Dimension 2")
                ax4.set_zlabel("Dimension 3")
            except Exception as e:
                logging.warning(f"Error in t-SNE/PCA 3D visualization: {str(e)}")
                ax4.text(0, 0, 0, "Error in 3D visualization", horizontalalignment='center')
        else:
            ax4.text(0, 0, 0, "3D visualization requires at least 4 samples and 3 features", 
                   horizontalalignment='center')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logging.error(f"Error plotting features: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting features: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig

def plot_confusion_matrix(cm):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : ndarray
        Confusion matrix
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set labels
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        # Set tick labels
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(['Non-seizure', 'Seizure'])
        ax.set_yticklabels(['Non-seizure', 'Seizure'])
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting confusion matrix: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    roc_auc : float
        Area under the curve
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        # Fill area under the curve
        ax.fill_between(fpr, tpr, alpha=0.2, color='b')
        
        # Set labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        
        # Set limits and grid
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True)
        
        # Add legend
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logging.error(f"Error plotting ROC curve: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting ROC curve: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig

def plot_brain_activity(data, ch_names, predictions=None):
    """
    Plot brain activity patterns
    
    Parameters:
    -----------
    data : ndarray
        EEG data with shape (n_channels, n_samples)
    ch_names : list
        Channel names
    predictions : dict
        Dictionary containing model predictions
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    """
    try:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Define layout
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Channel average activity
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Calculate average activity per channel
        mean_activity = np.mean(np.abs(data), axis=1)
        
        # Sort channels by activity
        sorted_indices = np.argsort(mean_activity)[::-1]
        sorted_ch_names = [ch_names[i] for i in sorted_indices]
        sorted_activity = mean_activity[sorted_indices]
        
        # Plot bar chart of channel activity
        bars = ax1.barh(np.arange(len(sorted_ch_names)), sorted_activity, color='skyblue', edgecolor='navy')
        
        # Set labels and title
        ax1.set_yticks(np.arange(len(sorted_ch_names)))
        ax1.set_yticklabels(sorted_ch_names)
        ax1.set_xlabel('Average Activity (μV)')
        ax1.set_title('Channel Activity Ranking')
        ax1.grid(True, axis='x')
        
        # Frequency spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate average PSD across channels
        from scipy import signal
        
        # Use a subset of channels (top 5 by activity) to avoid clutter
        top_channels = sorted_indices[:5]
        
        # Calculate and plot PSDs
        for idx in top_channels:
            freqs, psd = signal.welch(data[idx], fs=100, nperseg=256)  # Assuming 100 Hz sampling rate
            ax2.semilogy(freqs, psd, label=ch_names[idx])
        
        # Mark frequency bands
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 70)
        }
        
        for band, (fmin, fmax) in bands.items():
            ax2.axvspan(fmin, fmax, alpha=0.2, color=plt.cm.tab10(list(bands.keys()).index(band)))
            # Add text label at the center of each band
            ax2.text((fmin + fmax) / 2, ax2.get_ylim()[0] * 10, band, 
                   horizontalalignment='center', verticalalignment='bottom', fontsize=8)
        
        # Set labels and title
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density (μV²/Hz)')
        ax2.set_title('Frequency Spectrum (Top 5 Channels)')
        ax2.set_xlim([0, 70])  # Limit to 70 Hz
        ax2.legend()
        ax2.grid(True)
        
        # Time-frequency plot (spectrogram) for most active channel
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Get most active channel
        most_active_ch = sorted_indices[0]
        
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(data[most_active_ch], fs=100, nperseg=128, noverlap=64)
        
        # Plot spectrogram
        pcm = ax3.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(pcm, ax=ax3, label='Power/Frequency (dB/Hz)')
        
        # Set labels and title
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title(f'Spectrogram - Channel {ch_names[most_active_ch]}')
        ax3.set_ylim([0, 70])  # Limit to 70 Hz
        
        # Epileptic activity prediction visualization
        ax4 = fig.add_subplot(gs[1, 1])
        
        if predictions is not None:
            # Convert predictions to visualization
            # For example, a scrolling view of the most active channel with seizure predictions
            
            # Get predictions and true values
            y_pred = predictions.get('y_pred', None)
            y_true = predictions.get('y_test', None)
            
            if y_pred is not None:
                # Assuming we have windows of data that correspond to predictions
                # Create a pseudo-timeline
                timeline = np.arange(len(y_pred))
                
                # Plot prediction probabilities if available
                y_pred_proba = predictions.get('y_pred_proba', None)
                if y_pred_proba is not None:
                    ax4.plot(timeline, y_pred_proba, 'b-', alpha=0.7, label='Seizure Probability')
                
                # Mark predicted seizures
                seizure_markers = timeline[y_pred == 1]
                if len(seizure_markers) > 0:
                    ax4.scatter(seizure_markers, np.ones_like(seizure_markers), color='red', marker='o', 
                              s=100, label='Predicted Seizure', zorder=3)
                
                # Mark true seizures if available
                if y_true is not None:
                    true_seizure_markers = timeline[y_true == 1]
                    if len(true_seizure_markers) > 0:
                        ax4.scatter(true_seizure_markers, np.ones_like(true_seizure_markers) * 0.9, color='green', 
                                  marker='s', s=100, label='True Seizure', zorder=2)
                
                # Set labels and title
                ax4.set_xlabel('Window Index')
                ax4.set_ylabel('Seizure Probability')
                ax4.set_title('Seizure Prediction Timeline')
                ax4.set_ylim([0, 1.1])
                ax4.legend()
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, "No prediction data available", horizontalalignment='center', verticalalignment='center')
        else:
            # Alternative visualization: channel activity over time
            # Pick a few top channels
            top_5_channels = sorted_indices[:5]
            
            # Create a simplified time vector
            t = np.arange(data.shape[1]) / 100  # Assuming 100 Hz
            t = t[:min(10000, len(t))]  # Limit to first 100 seconds or less
            
            # Plot activity for top channels (first 100 seconds or less)
            for i, ch_idx in enumerate(top_5_channels):
                # Normalize and offset signal for better visualization
                signal = data[ch_idx, :len(t)]
                signal = signal - np.mean(signal)
                signal = signal / (np.max(np.abs(signal)) * 2) + i  # Scale and stack
                
                ax4.plot(t, signal, label=ch_names[ch_idx])
            
            # Set labels and title
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Channel')
            ax4.set_title('Top 5 Channels Activity')
            ax4.set_yticks(np.arange(len(top_5_channels)))
            ax4.set_yticklabels([ch_names[idx] for idx in top_5_channels])
            ax4.grid(True)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logging.error(f"Error plotting brain activity: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting brain activity: {str(e)}", horizontalalignment='center', verticalalignment='center')
        return fig
