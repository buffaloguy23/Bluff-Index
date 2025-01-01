#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 07:58:25 2024
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## IMPORT PACKAGES

# standard packages
import pandas as pd
import numpy as np
import json
import warnings
import random
import os

# plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ml packages
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import umap # pip install umap-learn
import hdbscan
from sklearn.ensemble import RandomForestClassifier

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## DEFINE FUNCTIONS

def normalize_coordinates(x, y, field_size=(64, 20)):
    """
    Normalize coordinates to fit within the field size, handling negative y values
    x: horizontal coordinate (0-53.3 yards)
    y: vertical coordinate (can be negative for pre-snap motion)
    """
    # Normalize x to fit within field width (0-64)
    norm_x = (x / 53.3) * field_size[0]
    
    # Shift y to handle negative values (-10 to 10 yards becomes 0 to 20)
    # Assuming pre-snap motion stays within 10 yards behind line of scrimmage
    norm_y = (y + 10) * (field_size[1] / 20)
    
    return norm_x, norm_y

def create_presnap_image(player_routes, field_size=(64, 20), color=True):
    if color == True:
        # RBG IMAGE
        image = np.zeros((field_size[1], field_size[0], 3)) # RGB image
        
        # Create a custom colormap for the gradient
        colors = ['blue', 'green', 'red']  # Start color, middle color, end color
        n_bins = 100  # Number of color gradations
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        for route in player_routes:
            route_length = len(route)
            for i, (x, y, _, _) in enumerate(route):
                # Normalize coordinates to field size
                norm_x, norm_y = normalize_coordinates(x, y, field_size)
                
                # Convert to integers for pixel coordinates
                pixel_x, pixel_y = int(norm_x), int(norm_y)
                
                
                # Check if coordinates are within bounds
                if 0 <= pixel_x < field_size[0] and 0 <= pixel_y < field_size[1]:
                    # Calculate color based on position in route
                    color = cmap(i / (route_length - 1))[:3]  # Get RGB values
                    image[pixel_y, pixel_x] = color
    else:
        image = np.zeros((field_size[1], field_size[0], 1))  # BW image    
        for route in player_routes:
            for i, (x, y, _, _) in enumerate(route):
                # Normalize coordinates to field size
                norm_x, norm_y = normalize_coordinates(x, y, field_size)
                
                # Convert to integers for pixel coordinates
                pixel_x, pixel_y = int(norm_x), int(norm_y)
                
                # Check if coordinates are within bounds
                if 0 <= pixel_x < field_size[0] and 0 <= pixel_y < field_size[1]:
                    image[pixel_y, pixel_x] = 1  # For binary
    
    return image

def create_presnap_images(play_data, field_size=(64, 20), color=True):
    """
    Create images for all plays with pre-snap motion
    Returns: numpy array of images
    """
    play_images = []
    for play in play_data:
        if play['presnap_motion'] == True:
            routes = play['routes']
            
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            image = create_presnap_image(routes, field_size, color)
            play_images.append({
                'image': image,
                'metadata': play['metadata'],
            })
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    # Convert to numpy array just for the images
    images_array = np.array([play['image'] for play in play_images])
    metadata_list = [x['metadata'] for x in play_images]
    
    return images_array, metadata_list

def set_random_seeds(seed_value=42):
    """
    Set random seeds for reproducibility across all libraries used
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # For reproducible GPU operations
    try:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except:
        pass

class Autoencoder(Model):
    def __init__(self, latent_dim, input_shape, seed=42):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Calculate the dimensions after each stride
        h, w = input_shape[0], input_shape[1]
        h1, w1 = h // 2, w // 2  # First stride
        h2, w2 = h1 // 2, w1 // 2  # Second stride
        
        # Add kernel_initializer with fixed seed for reproducibility
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer=initializer),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer=initializer),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu', kernel_initializer=initializer),
        ])
        
        # Calculate flattened dimension for decoder's initial dense layer
        self.flatten_dim = h2 * w2 * 8
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(self.flatten_dim, activation='relu', kernel_initializer=initializer),
            layers.Reshape((h2, w2, 8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', kernel_initializer=initializer),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same', kernel_initializer=initializer),
            layers.Conv2D(input_shape[2], kernel_size=(3, 3), activation='sigmoid', padding='same', kernel_initializer=initializer)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

def train_autoencoder(tr_images, input_shape, latent_dim, epochs, seed=42, verbose=False, plot_flag=False, early_stop=True,):
    # Set random seeds
    set_random_seeds(seed)
    
    # Initialize autoencoder
    autoencoder = Autoencoder(latent_dim, input_shape, seed=seed)

    # Use a fixed seed for the optimizer
    optimizer = tf.keras.optimizers.Adam()
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=verbose
    )

    # Train the model
    if early_stop:
        history = autoencoder.fit(
            tr_images,
            tr_images,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=verbose,
        )
    else:
        history = autoencoder.fit(
            tr_images,
            tr_images,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.2,
            verbose=verbose,
        )
    
    # After training, you can analyze the training history
    if plot_flag: 
        plt.plot(history.history['loss'][1:], label='Training Loss')
        plt.plot(history.history['val_loss'][1:], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    
    return autoencoder

def apply_UMAP(df, features, N=3):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*")  # For the first warning
        warnings.filterwarnings("ignore", message="n_neighbors is larger.*")  # For the second warning
        scaler = StandardScaler()
        data = scaler.fit_transform(df[features])
        reducer = umap.UMAP(n_components=N, random_state=42)
        embedding = reducer.fit_transform(data)
        emb_df = pd.DataFrame(embedding, columns=['pc'+str(x+1) for x in range(N)])
        return pd.concat([df, emb_df], axis=1), scaler, reducer

def get_dbscan_label(data_df, features, mcs=15):
    # Train DBSCAN on all data
    clustering_model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1)
    labels = clustering_model.fit_predict(data_df[features])  
    # Train vanilla RFC to put outliers (-1) into most similar buckets
    tr_mask = pd.Series(labels).astype(str) != "-1"
    te_mask = pd.Series(labels).astype(str) == "-1"
    if te_mask.sum() > 0:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(data_df.loc[tr_mask,features], labels[tr_mask])
        labels[te_mask] = clf.predict(data_df.loc[te_mask,features])
    # Return labels
    return labels

def plot_play(play_data):
    fig, ax = plt.subplots(figsize=(9, 6))  # Vertical field
    
    max_y = []
    
    # Set field boundaries
    ax.set_xlim(0, 53.3)
    ax.set_ylim(-20, 0)
    
    # Draw field lines
    for y in range(0, 101, 10):
        ax.axhline(y, color='gray', linestyle='--', alpha=0.3)
    
    # Plot routes
    for i, route in enumerate(play_data['routes']):
        print('here')
        x, y, vx, vy = zip(*route)
        max_y += y
        
        # Calculate speed at each point
        speeds = np.array(list(range(len(route))))
        # speeds = np.sqrt(np.array(vx)**2 + np.array(vy)**2)
        
        # Create a set of line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from speed to colors
        norm = Normalize(vmin=speeds.min(), vmax=speeds.max())
        
        # Create a line collection
        lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=10)
        lc.set_linewidth(5)
        ax.add_collection(lc)
        plt.draw()
        
        # lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(speeds)
        lc.set_linewidth(5)
        line = ax.add_collection(lc)
        
    # Add colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Speed (yards/second)')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Set labels
    ax.set_xlabel('Width (yards)')
    ax.set_ylabel('Length (yards)')
    ax.set_title(f"Play Visualization\nDown: {play_data['metadata']['down']}, Distance: {play_data['metadata']['distance']}")
    
    # Add metadata as text
    metadata_text = "\n".join([f"{k}: {v}" for k, v in play_data['metadata'].items()])
    plt.figtext(0.02, 0.02, metadata_text, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

def run(
    save_root='/Users/danbickelhaupt/Desktop/coding_projects/NFL Big Data Bowl 2025/curated_data/', 
    presnap_filename='processed_presnap_motion_data.json',
    image_color_flag=True,
    early_stop=True,
    plot_flag=False,
    epochs = 100,
    seed=42
    ):
    
    # Set random seeds at the start
    set_random_seeds(seed)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## LOAD & PROCESS DATA
    
    # Create filepath
    presnap_file_path = save_root + presnap_filename
    
    print('Loading presnap data...')
    # Load pre-snap data from the JSON file
    with open(presnap_file_path, 'r') as f:
        presnap_data = json.load(f)
    # Create images for all presnap plays
    presnap_images, presnap_metadata = create_presnap_images(
        play_data=presnap_data,
        field_size=(64,20),
        color=image_color_flag
        )
    presnap_metadata = pd.DataFrame(presnap_metadata)
    presnap_metadata['id'] = presnap_metadata['gameId'].astype(str) + presnap_metadata['playId'].astype(str)
    print('\tPresnap data loaded.')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## ADD PRE-SNAP MOTION LABEL TO DATASET
    
    tr_mask_presnap = presnap_metadata['week'] <= 9
    te_mask_presnap = presnap_metadata['week'] > 10
    
    # Create & train autoencoder
    print("\nBUILDING PRE-SNAP AUTOENCODER...")
    presnap_image_size = (20, 64, 3) if image_color_flag == True else (20, 64, 1)
    latent_layer_size = 32
    verbose = True
    presnap_autoencoder = train_autoencoder(
        presnap_images[tr_mask_presnap], 
        presnap_image_size, 
        latent_layer_size, 
        epochs, 
        seed=seed,
        verbose=verbose, 
        plot_flag=plot_flag, 
        early_stop=early_stop
        )
    
    # Encode training data and use to train UMAP model
    encoded_plays_tr = pd.DataFrame(
        presnap_autoencoder.encode(presnap_images[tr_mask_presnap]), 
        columns=['d'+str(x) for x in range(latent_layer_size)])
    _, std_scaler, presnap_umapper = apply_UMAP(encoded_plays_tr, encoded_plays_tr.columns, N=3)
    
    # Extract UMAP dimensional reduction for train and test data
    presnap_metadata = pd.concat([
        presnap_metadata,
        pd.DataFrame(presnap_umapper.transform(presnap_autoencoder.encode(presnap_images)), columns=['pc1','pc2','pc3']).astype(float)
        ], axis=1)
    
    # Assign labels to training set, fill -1 outliers with RFC output
    presnap_metadata.loc[tr_mask_presnap,'presnap_label'] = get_dbscan_label(presnap_metadata[tr_mask_presnap], ['pc1','pc2','pc3'], mcs=10)
    
    # Assign labels to test set using trained RFC
    if te_mask_presnap.sum() > 0:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(presnap_metadata.loc[tr_mask_presnap,['pc1','pc2','pc3']], presnap_metadata.loc[tr_mask_presnap,'presnap_label'])
        presnap_metadata.loc[te_mask_presnap,'presnap_label'] = clf.predict(presnap_metadata.loc[te_mask_presnap,['pc1','pc2','pc3']])
    
    if plot_flag:

        ## CODE TO VISUALIZE CLUSTERS
        search_id = '202209080080'
        
        # Plot play
        for item in presnap_data:
            if (item['metadata']['gameId']==2022090800) & (item['metadata']['playId']==80):
                yy = item
                break
        plot_play(yy)
        
        # Get play from various places
        r = presnap_metadata[presnap_metadata['id']==search_id].index[0]
        label_ = presnap_metadata.loc[r,'presnap_label']
        x1 = np.mean(presnap_images[presnap_metadata.index[presnap_metadata['presnap_label']==label_]], axis=0)#.reshape(presnap_image_size[:-1])
        
        # Get recreated image
        x2 = presnap_autoencoder.decode(presnap_autoencoder.encode(presnap_images[r].reshape((1,20,64,3))))
        x2 = np.array(x2).reshape(20,64,3)
        
        # Make play vs cluster plots
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(presnap_images[r].squeeze(), cmap='gray', origin='lower')
        plt.title(f"presnap label = {label_} {r}")
        
        plt.subplot(1,3,2)
        plt.imshow(x1.squeeze()*10, cmap='gray', origin='lower')
        plt.title(f"presnap label = {label_} - avg")
        
        plt.subplot(1,3,3)
        plt.imshow(x2.squeeze()*10, cmap='gray', origin='lower')
        plt.title(f"presnap label = {search_id} recreation")
        plt.show()
        
        plt.figure()
        plt.subplot(1,3,1)
        sns.heatmap(np.sum(presnap_images[r],axis=-1))
        plt.title(f"presnap label = {label_} (r)")
        
        plt.subplot(1,3,2)
        sns.heatmap(np.sum(x1,axis=-1))
        plt.title(f"presnap label = {label_} - avg")
        
        plt.subplot(1,3,3)
        sns.heatmap(np.sum(x2,axis=-1))
        plt.title(f"presnap label = {label_} (r) - recreated")
        plt.show()
    
    print("\tPRE-SNAP AUTOENCODER BUILD COMPLETE.")
    
    return presnap_metadata

