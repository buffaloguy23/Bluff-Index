#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:46:25 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO

def create_nfl_logo_correlation_plot(predictions, actuals, team_abbrs, 
                                   base_logo_path, img_type='.png',
                                   logo_display_size=(50, 50),
                                   title="Team Performance Correlation",
                                   xlabel='PREDICTION',
                                   ylabel='ACTUAL'):
    """
    Create a modern-looking correlation plot using NFL team logos as data points.
    
    Parameters:
    -----------
    predictions : array-like
        The predicted values
    actuals : array-like
        The actual values
    team_abbrs : list
        List of team abbreviations corresponding to each data point
    base_logo_path : str
        Base path to the logo directory
    img_type : str, optional
        Image file extension (default: '.png')
    logo_display_size : tuple, optional
        Desired size for logos in pixels (width, height)
    title : str, optional
        The title of the plot
    """
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    TITLE_SIZE = 20

    # Update font sizes globally
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=TITLE_SIZE)

    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate statistics
    r = np.corrcoef(predictions, actuals)[0,1]
    r_squared = r**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, actuals)
    
    # Create figure with specific size and DPI
    # fig = plt.figure(figsize=(12, 10), dpi=100)
    fig = plt.figure(figsize=(8, 5))
    
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create main axis
    ax = plt.gca()
    
    # Plot trend line first (so it's behind the logos)
    x_range = np.array([min(predictions), max(predictions)])
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, color='#dc3545', linestyle='--', linewidth=2.5, label='Trend Line')
    
    def get_logo_image(team_abbr):
        """Helper function to load and process team logo with standardized size"""
        try:
            # Construct full logo path
            logo_path = f"{base_logo_path}/{team_abbr.lower()}{img_type}"
            img = Image.open(logo_path)
            
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize image to standard size while maintaining aspect ratio
            img.thumbnail(logo_display_size, Image.Resampling.LANCZOS)
            
            # Create new image with standard size and paste resized logo centered
            standardized_img = Image.new('RGBA', logo_display_size, (0, 0, 0, 0))
            paste_x = (logo_display_size[0] - img.size[0]) // 2
            paste_y = (logo_display_size[1] - img.size[1]) // 2
            standardized_img.paste(img, (paste_x, paste_y), img)
            
            return standardized_img
            
        except Exception as e:
            print(f"Failed to load logo for {team_abbr}: {str(e)}")
            # Create text-based placeholder
            fig_temp = plt.figure(figsize=(2, 2))
            plt.text(0.5, 0.5, team_abbr, fontsize=14, ha='center', va='center')
            plt.axis('off')
            canvas = fig_temp.canvas
            buffer = BytesIO()
            canvas.print_png(buffer)
            plt.close(fig_temp)
            buffer.seek(0)
            img = Image.open(buffer)
            img = img.resize(logo_display_size, Image.Resampling.LANCZOS)
            return img
    
    # Calculate zoom factor based on plot size and desired logo size
    fig_width_inches = fig.get_size_inches()[0]
    fig_width_pixels = fig_width_inches * fig.dpi
    x_range = max(predictions) - min(predictions)
    zoom_factor = (logo_display_size[0] / fig_width_pixels) * (x_range / 0.1)
    
    # Add team logos as scatter points
    for x, y, team in zip(predictions, actuals, team_abbrs):
        # Load and process logo
        logo = get_logo_image(team)
        
        # Create OffsetImage with standardized zoom
        imagebox = OffsetImage(logo, zoom=0.5) #zoom_factor)
        imagebox.image.axes = ax
        
        # Create and add AnnotationBbox
        ab = AnnotationBbox(imagebox, (x, y),
                          frameon=False,
                          pad=0,
                          box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    
    # Customize grid
    ax.grid(True, linestyle='-', alpha=0.3, color='#212529')
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    # Add statistics text with larger font
    stats_text = f'R = {r:.4f}\nR² = {r_squared:.4f}\np-value = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=MEDIUM_SIZE,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=10))
    
    # Customize axis labels
    ax.set_xlabel(xlabel, fontsize=MEDIUM_SIZE, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=MEDIUM_SIZE, fontweight='bold', labelpad=15)
    
    # Set title with custom styling
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
    
    # # Add hover labels for teams
    # for x, y, team in zip(predictions, actuals, team_abbrs):
    #     ax.annotate(team,
    #                 (x, y),
    #                 xytext=(8, 8),
    #                 textcoords='offset points',
    #                 fontsize=SMALL_SIZE,
    #                 fontweight='bold',
    #                 alpha=0.8,
    #                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Add after plotting logos but before tight_layout()
    margin = 0.05  # Add 5% margin around the data points
    x_min, x_max = min(predictions), max(predictions)
    y_min, y_max = min(actuals), max(actuals)
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_modern_correlation_plot(predictions, actuals, title="Prediction vs Actual Correlation", xlabel="PREDICTION", ylabel="ACTUAL"):
    """
    Create a modern-looking correlation plot with statistics.
    
    Parameters:
    -----------
    predictions : array-like
        The predicted values
    actuals : array-like
        The actual values
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    
    # Calculate statistics
    r = np.corrcoef(predictions, actuals)[0,1]
    r_squared = r**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, actuals)
    
    # Create figure with specific size and DPI
    fig = plt.figure(figsize=(6, 3), dpi=75)
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    TITLE_SIZE = 18

    # Update font sizes globally
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)   # fontsize of the figure title
    
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create main axis
    ax = plt.gca()
    
    # Plot data points
    ax.scatter(predictions, actuals, color='#0d6efd', s=100, alpha=0.7, label='Data Points')
    
    # Plot trend line
    x_range = np.array([min(predictions), max(predictions)])
    y_range = slope * x_range + intercept
    ax.plot(x_range, y_range, color='#dc3545', linestyle='--', linewidth=2, label='Trend Line')
    
    # Customize grid
    ax.grid(True, linestyle='-', alpha=0.2, color='#212529')
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    # Add statistics text
    stats_text = f'R = {r:.4f}\nR² = {r_squared:.4f}\np-value = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Customize axis labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=10)
    
    # Set title with custom styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # # Add legend
    # ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_calibration_curve(y_test, y_prob, n_bins, verbose=True):
    # Get calibration metrics (for later)
    cal_metrics = compute_calibration_accuracy(y_test, y_prob, n_bins=n_bins)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=n_bins,
        strategy='uniform'
    )
    
    # Create the calibration plot
    plt.figure(figsize=(10, 6))  # Increased figure size
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    TITLE_SIZE = 18

    # Update font sizes globally
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)   # fontsize of the figure title
    
    # Plot with thicker lines
    plt.plot(prob_pred, prob_true, marker='o', markersize=10, linewidth=2.5, label='Random Forest')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2.5, label='Perfectly Calibrated')
    
    # Set labels and title with increased font sizes
    plt.xlabel('Mean Predicted Probability', fontsize=LARGE_SIZE, labelpad=10)
    plt.ylabel('Fraction of Positives', fontsize=LARGE_SIZE, labelpad=10)
    plt.title('Calibration Curve - Random Forest Classifier', fontsize=TITLE_SIZE, pad=20)
    
    # Enhanced legend
    plt.legend(fontsize=LARGE_SIZE, frameon=True, facecolor='white', edgecolor='gray', 
              framealpha=0.9, loc='upper left')
    
    # Enhanced grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics text with increased font size and better positioning
    plt.text(0.52, 0.05,  # Adjusted position
             f"Weighted Calibration Accuracy: {cal_metrics['weighted_accuracy']:.3f}\n"
             f"Unweighted Calibration Accuracy: {cal_metrics['unweighted_accuracy']:.3f}",
             transform=plt.gca().transAxes,
             fontsize=LARGE_SIZE,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=10))
    
    # Increase tick label sizes
    plt.xticks(fontsize=MEDIUM_SIZE)
    plt.yticks(fontsize=MEDIUM_SIZE)
    
    # Add some padding to the layout
    plt.tight_layout()
    
    plt.show()
    
    # Print metrics if required
    if verbose:
        print("\nCalibration Metrics:".upper())
        print(f"Weighted Calibration Accuracy: {cal_metrics['weighted_accuracy']:.3f}")
        print(f"Unweighted Calibration Accuracy: {cal_metrics['unweighted_accuracy']:.3f}")
        print("\nBin Statistics:".upper())
        print("Bin Counts:", cal_metrics['bin_counts'])
        print("True Probabilities:", np.round(cal_metrics['bin_true_probs'], 3))
        print("Predicted Probabilities:", np.round(cal_metrics['bin_pred_probs'], 3))
        print(f"Brier Score Loss: {brier_score_loss(y_test, y_prob):.4f}")

def plot_bisector(pred, actual, title_=''):
    '''
    Function to plot predictions vs actuals on a standard bisector chart.
    '''    
    slope, intercept, r_value, p_value, std_err = stats.linregress(pred, actual)
    x = np.array([np.min(pred), np.max(pred)])
 
    val = [np.min(actual), np.max(actual)]
    plt.figure()
    # plt.plot(val,val,'k--')
    plt.scatter(pred, actual)
    plt.plot(x,slope*x + intercept, 'r--')
    plt.grid('minor')
    plt.xlabel('PREDICTION')
    plt.ylabel('ACTUAL')
    plt.title(title_)
    plt.text(0.15*(np.max(pred) - np.min(pred)) + np.min(pred), np.max(actual)-0.10*(np.max(actual) - np.min(actual)), f"R={r_value:0.4f}\nR^2={r_value**2:0.4f}\np-value={p_value:0.4f}",horizontalalignment='center',verticalalignment='center')#, transform=ax[0].transAxes) 
    plt.show()


def get_feature_importance_with_ci(model, n_iterations=100):
   # Collect importances
   importances = []
   for tree in model.estimators_:
       importances.append(tree.feature_importances_)
   importances = np.array(importances)
   # Calculate mean and confidence intervals
   importance_df = pd.DataFrame({
       'feature': model.feature_names_in_,
       'importance': model.feature_importances_,
       'importance_std': np.std(importances, axis=0),
       'importance_ci_lower': np.percentile(importances, 2.5, axis=0),
       'importance_ci_upper': np.percentile(importances, 97.5, axis=0)
   })
   # Sort by importance
   importance_df = importance_df.sort_values('importance', ascending=False)
   importance_df = importance_df.reset_index(drop=True)
   # Return results
   return importance_df

def encode_categorical_columns(df, columns_to_encode, verbose=True):
    # Dictionary to store the label encoders
    encoders = {}
    df_encoded = df.copy()
    for column in columns_to_encode:
        # Create and fit label encoder for each column
        encoders[column] = LabelEncoder()
        df_encoded[f'{column}'] = encoders[column].fit_transform(df_encoded[column])
        # Create and display mapping for this column
        mapping = dict(zip(encoders[column].classes_, encoders[column].transform(encoders[column].classes_)))
        if verbose:
            print(f"\nMapping for {column}:")
            print(mapping)
    return df_encoded, encoders

def reverse_encoding(df, encoders, columns_to_decode):
    df_decoded = df.copy()
    for column in columns_to_decode:
        encoded_column = f'{column}_encoded'
        if encoded_column in df_decoded.columns:
            original_values = encoders[column].inverse_transform(df_decoded[encoded_column])
            df_decoded[f'{column}_decoded'] = original_values
    return df_decoded

def compute_calibration_accuracy(y_true, y_prob, n_bins=10):
    """
    Compute weighted and unweighted calibration accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for probability buckets
        
    Returns:
    --------
    dict with calibration metrics
    """
    # Create probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    # Initialize arrays for storing metrics
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    
    # Compute frequencies for each bin
    for bin_idx in range(n_bins):
        mask = binids == bin_idx
        bin_total[bin_idx] = mask.sum()
        if bin_total[bin_idx] > 0:
            bin_true[bin_idx] = y_true[mask].sum()
            bin_sums[bin_idx] = y_prob[mask].sum()
    
    # Calculate average predicted probability in each bin
    bin_pred = np.zeros(n_bins)
    nonzero_bins = bin_total > 0
    bin_pred[nonzero_bins] = bin_sums[nonzero_bins] / bin_total[nonzero_bins]
    
    # Calculate actual probability in each bin
    bin_true_prob = np.zeros(n_bins)
    bin_true_prob[nonzero_bins] = bin_true[nonzero_bins] / bin_total[nonzero_bins]
    
    # Calculate calibration errors
    errors = np.abs(bin_pred - bin_true_prob)
    
    # Compute unweighted calibration accuracy
    unweighted_acc = 1 - np.mean(errors[nonzero_bins])
    
    # Compute weighted calibration accuracy
    weights = bin_total / bin_total.sum()
    weighted_acc = 1 - np.sum(errors * weights)
    
    return {
        'unweighted_accuracy': unweighted_acc,
        'weighted_accuracy': weighted_acc,
        'bin_counts': bin_total,
        'bin_true_probs': bin_true_prob,
        'bin_pred_probs': bin_pred
    }

def compute_cdf_distance(data_motion, data_no_motion, n_points=100):
    """
    Compute the signed distance between two CDF curves.
    Positive distance indicates motion curve is to the right of no_motion curve.
    
    Parameters:
    -----------
    data_motion : array-like
        Data for the motion condition
    data_no_motion : array-like
        Data for the no-motion condition
    n_points : int
        Number of points to evaluate distance at
    
    Returns:
    --------
    dict containing:
        'y_values': y-coordinates where distances were computed
        'distances': signed distances at each y-value
        'mean_distance': average signed distance
        'median_distance': median signed distance
        'max_distance': maximum absolute distance
    """
    # Compute CDFs for both datasets
    counts_motion, bins_motion = np.histogram(data_motion, bins=50, density=True)
    cdf_motion = np.cumsum(counts_motion) * np.diff(bins_motion)
    cdf_motion = np.insert(cdf_motion, 0, 0)  # Add 0 at the start
    
    counts_no_motion, bins_no_motion = np.histogram(data_no_motion, bins=50, density=True)
    cdf_no_motion = np.cumsum(counts_no_motion) * np.diff(bins_no_motion)
    cdf_no_motion = np.insert(cdf_no_motion, 0, 0)  # Add 0 at the start
    
    # Create interpolation functions for both CDFs
    f_motion = interpolate.interp1d(cdf_motion, bins_motion, 
                                  bounds_error=False, fill_value=(bins_motion[0], bins_motion[-1]))
    f_no_motion = interpolate.interp1d(cdf_no_motion, bins_no_motion, 
                                     bounds_error=False, fill_value=(bins_no_motion[0], bins_no_motion[-1]))
    
    # Evaluate distances at evenly spaced points along y-axis (CDF values)
    y_values = np.linspace(0.01, 0.99, n_points)  # Avoid 0 and 1 for numerical stability
    
    # Compute x-values for both curves at each y-value
    x_motion = f_motion(y_values)
    x_no_motion = f_no_motion(y_values)
    
    # Compute signed distances (positive when motion curve is to the right)
    distances = x_motion - x_no_motion
    
    return {
        'y_values': y_values,
        'distances': distances,
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'max_distance': np.max(np.abs(distances))
    }