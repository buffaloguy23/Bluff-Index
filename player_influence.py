#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:37:47 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

class PlayerInfluence:
    def __init__(self, base_std=5.0, speed_factor=0.3, max_speed=10.0):
        self.base_std = base_std
        self.speed_factor = speed_factor
        self.max_speed = max_speed
        
    def get_covariance(self, speed, direction):
        """
        Get covariance matrix adjusted for player speed and direction
        Returns 2x2 covariance matrix
        """
        # Normalize speed
        speed_ratio = min(speed / self.max_speed, 1.0)
        
        # Convert direction to radians
        theta = np.radians(direction)
        
        # Create rotation matrix
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        
        # Create scaling matrix - elongate in direction of movement
        scaling = np.array([[1 + speed_ratio * self.speed_factor, 0],
                          [0, 1 - speed_ratio * self.speed_factor/2]])
        
        # Compute covariance matrix
        base_cov = np.eye(2) * self.base_std**2
        return rotation @ scaling @ base_cov @ scaling @ rotation.T
    
    def get_influence(self, X, Y, player_x, player_y, speed, direction):
        """
        Calculate player's influence across a grid of points
        
        Args:
            X: 2D array of x coordinates from meshgrid
            Y: 2D array of y coordinates from meshgrid
            player_x: player's x coordinate (scalar)
            player_y: player's y coordinate (scalar)
            speed: player's speed (scalar)
            direction: player's direction (scalar)
            
        Returns:
            2D array of influence values matching X,Y dimensions
        """
        # Stack coordinates into position array
        positions = np.dstack([X, Y])  # Shape: (n_y, n_x, 2)
        
        # Get mean and covariance
        mean = np.array([player_x, player_y])
        cov = self.get_covariance(speed, direction)
        
        # Reshape positions for vectorized calculation
        positions_2d = positions.reshape(-1, 2)
        
        # Calculate Gaussian for all points
        influence = multivariate_normal.pdf(positions_2d, mean=mean, cov=cov)
        
        # Reshape back to match input grid
        influence = influence.reshape(X.shape)
        
        # Normalize by the value at player's position
        player_influence = multivariate_normal.pdf(mean, mean=mean, cov=cov)
        influence = influence / player_influence
        
        return influence

class MotionAnalyzer:
    def __init__(self, tracking_data, plays_data, player_play_data):
        self.tracking = tracking_data
        self.plays = plays_data 
        self.player_play = player_play_data
        self.influence_model = PlayerInfluence()
        
        # Create standard grid for field
        self.x = np.linspace(0, 120, 120)  # 1 yard resolution
        self.y = np.linspace(0, 53.3, 53)  # 1 yard resolution
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def get_motion_plays(self):
        """
        Get plays where at least one player is in motion pre-snap
        """
        # motion_players = self.player_play[self.player_play['inMotionAtBallSnap'] == True]
        # return motion_players[['gameId', 'playId']].drop_duplicates()
        
        # Assume pre-processing was correctly performed such that only plays in tracking_df are pre-snap motion ones
        return self.tracking[['gameId', 'playId']].drop_duplicates()
    
    def get_pre_snap_frames(self, game_id, play_id):
        """
        Get tracking data frames before the snap
        """
        play_tracking = self.tracking[
            (self.tracking['gameId'] == game_id) & 
            (self.tracking['playId'] == play_id)
        ]
        
        # Get snap frame
        snap_frame = play_tracking['frameId'].max() # preprocessing makes sure last frame is right before snap
        # play_tracking[play_tracking['event'] == 'ball_snap']['frameId'].iloc[0]
        
        # Get frames just before motion starts and right before snap
        pre_motion_idx = max([snap_frame - 30, 0])
        pre_snap_idx = snap_frame - 1
        pre_motion = play_tracking[play_tracking['frameId'] == pre_motion_idx]
        pre_snap = play_tracking[play_tracking['frameId'] == pre_snap_idx]
        
        return pre_motion, pre_snap, pre_motion_idx, pre_snap_idx
    
    def calculate_influence_surface(self, tracking_frame, possession_team):
        """
        Calculate influence surface for a single frame using vectorized operations
        """
        # Initialize influence surface
        influence = np.zeros_like(self.X)
        
        # Calculate each player's influence
        for _, player in tracking_frame.iterrows():
            if pd.isna(player['nflId']):  # Skip ball
                continue
                
            sign = 1 if player['club'] == possession_team else -1
            
            player_influence = self.influence_model.get_influence(
                self.X, self.Y, 
                player['x'], player['y'], 
                player['s'], player['dir']
            )
            
            influence += sign * player_influence
            
        return influence
    
    def analyze_play(self, game_id, play_id):
        """
        Analyze a single play's pre-snap motion
        """
        # Get tracking frames
        pre_motion, pre_snap, pre_motion_frameId, pre_snap_frameId = self.get_pre_snap_frames(game_id, play_id)
        
        # Get possession team
        possession_team = self.plays[
            (self.plays['gameId'] == game_id) & 
            (self.plays['playId'] == play_id)
        ]['possessionTeam'].iloc[0]
        
        # Calculate influence surfaces
        influence_pre = self.calculate_influence_surface(pre_motion, possession_team)
        influence_post = self.calculate_influence_surface(pre_snap, possession_team)
        
        # Calculate influence change
        influence_change = influence_post - influence_pre
        
        return {
            'pre_motion': influence_pre,
            'pre_snap': influence_post,
            'change': influence_change,
            'pre_motion_frameId':pre_motion_frameId,
            'pre_snap_frameId': pre_snap_frameId,
            'possessionTeam': possession_team,
        }
    
    def analyze_all_motion_plays(self):
        """
        Analyze all plays with pre-snap motion
        """
        results = {}
        motion_plays = self.get_motion_plays()
        
        for _, play in motion_plays.iterrows():
            game_id = play['gameId']
            play_id = play['playId']
            results[(game_id, play_id)] = self.analyze_play(game_id, play_id)
            
        return results
    
def plot_influence_surface(X, Y, influence, title, tracking_frame=None, possession_team=None):
    """
    Plot influence surface with player positions
    
    Args:
        X: 2D array of x coordinates from meshgrid
        Y: 2D array of y coordinates from meshgrid
        influence: 2D array of influence values
        title: Plot title
        tracking_frame: DataFrame containing player positions for one frame
        possession_team: String indicating possession team abbreviation
    """
    plt.figure(figsize=(15, 8))
    
    # Plot influence surface
    plt.contourf(X, Y, influence, levels=20, cmap='coolwarm',
                          vmin=-np.abs(influence).max(), vmax=np.abs(influence).max())
    plt.colorbar(label='Team Influence')
    
    # Add field boundaries
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    
    # Add yard lines
    for x in range(10, 110, 10):
        plt.axvline(x=x, color='white', linestyle='-', alpha=0.2)
    
    # Plot players if provided
    if tracking_frame is not None and possession_team is not None:
        for _, player in tracking_frame.iterrows():
            if pd.isna(player['nflId']):  # Skip ball
                continue
                
            # Determine marker based on offense/defense
            marker = 'X' if player['club'] == possession_team else 'o'
            color = 'lime' if player['club'] == possession_team else 'red'
            
            # Plot player position
            plt.plot(player['x'], player['y'], marker, color=color, markersize=10, 
                    markeredgewidth=2, label=player['club'] if marker == 'X' else '')
            
            # Add jersey numbers
            if not pd.isna(player['jerseyNumber']):
                plt.text(player['x'], player['y'] + 1, str(int(player['jerseyNumber'])),
                        color='white', ha='center', va='center', fontsize=8)
    
    # Add legend (only once for each team)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), 
              title='Possession', loc='upper right')
    
    plt.title(title)
    plt.xlabel('Field Position (yards)')
    plt.ylabel('Field Width (yards)')
    
    # Use field proportions
    plt.gca().set_aspect('equal')
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def process_tracking_files(tracking_dir, plays_df, player_play_df, games_df, weeks=range(1,10)):
    """
    Process tracking data files to extract all pre-snap motion data
    
    Args:
        tracking_dir: Directory containing tracking week files
        plays_df: Plays dataframe
        player_play_df: Player play dataframe with motion indicators
        weeks: Range of weeks to process (default 1-9)
        
    Returns:
        DataFrame containing all relevant pre-snap tracking data
    """
    # Merge in game info, create "id" field
    player_play_df = player_play_df.merge(how='left', on='gameId', right=games_df[['gameId','season','week']])
    player_play_df['id'] = player_play_df['gameId'].astype(str)+player_play_df['playId'].astype(str)
    
    # Essential columns to keep
    keep_cols = ['gameId', 'playId', 'frameId', 'time', 'event', 
                 'nflId', 'displayName', 'club', 'x', 'y', 's', 'dir',
                 'frameType','jerseyNumber']
    
    all_tracking_data = []
    
    for week in weeks:
        filename = os.path.join(tracking_dir, f'tracking_week_{week}.csv')
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping...")
            continue
            
        print(f"Processing week {week}...")
        
        # Read only required columns
        df = pd.read_csv(filename, usecols=keep_cols)
        
        # Filter to plays with motion
        mask = (player_play_df['week']==week) & (player_play_df['motionSinceLineset'] == True)
        player_play_df_now = player_play_df[mask].reset_index(drop=True)
        
        # Create "id" feature and filter to (1) only "id"s present in the metadata df & (2) frames prior to the snap
        df['id'] = df['gameId'].astype(str)+df['playId'].astype(str)
        mask = (df['id'].isin(player_play_df_now['id'].unique())) & (df['frameType']=='BEFORE_SNAP')
        df = df[mask]
        
        # Add motion player information
        motion_players = player_play_df[
            (player_play_df['motionSinceLineset'] == True)
        ][['gameId', 'playId', 'nflId']]
        
        def process_play_frames(play_df):
            """
            Process frames for a single play
            """
            play_motion_players = motion_players[
                (motion_players['gameId'] == play_df['gameId'].iloc[0]) &
                (motion_players['playId'] == play_df['playId'].iloc[0])
            ]['nflId'].tolist()
            
            # Find frames where motion players are moving
            motion_frames = play_df[
                (play_df['nflId'].isin(play_motion_players)) &
                (play_df['s'] > 0.1)  # Speed threshold to detect movement
            ]['frameId']
            
            if len(motion_frames) == 0:
                return pd.DataFrame()
                
            motion_start = motion_frames.min()
            
            # Keep frames from just before motion starts to end of pre-snap
            return play_df[play_df['frameId'] >= motion_start - 10]
        
        # Process frames for each play
        processed_frames = []
        for (game_id, play_id), play_df in df.groupby(['gameId', 'playId']):
            processed_play = process_play_frames(play_df)
            if not processed_play.empty:
                processed_frames.append(processed_play)
        
        if processed_frames:
            df = pd.concat(processed_frames, ignore_index=True)
            all_tracking_data.append(df)
    
    if not all_tracking_data:
        raise ValueError("No valid tracking data found")
        
    # Combine all weeks
    final_df = pd.concat(all_tracking_data, ignore_index=True)
    
    # Add possession team info
    play_possession = plays_df[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']]
    final_df = final_df.merge(play_possession, on=['gameId', 'playId'], how='left')
    
    print(f"Processed data shape: {final_df.shape}")
    return final_df

def get_motion_summary(tracking_df, player_play_df):
    """
    Generate summary statistics about the motion plays
    """
    # Count unique plays with motion
    n_plays = len(tracking_df[['gameId', 'playId']].drop_duplicates())
    
    # Get motion player info
    motion_players = player_play_df[player_play_df['motionSinceLineset'] == True]
    
    # Summarize motion by team
    team_motion = tracking_df.merge(
        motion_players[['gameId', 'playId', 'nflId', 'teamAbbr']],
        on=['gameId', 'playId', 'nflId']
    )
    team_counts = team_motion[['gameId', 'playId', 'teamAbbr']].drop_duplicates()['teamAbbr'].value_counts()
    
    # Get average motion duration
    motion_duration = tracking_df.groupby(['gameId', 'playId'])['frameId'].nunique().mean()
    
    print("\nMotion Play Summary:")
    print(f"Total plays with motion: {n_plays}")
    print(f"Average frames per motion: {motion_duration:.1f}")

    return {
        'n_plays': n_plays,
        'avg_duration': motion_duration,
        'team_counts': team_counts
    }

def load_and_prepare_data(data_dir):
    """
    Load and prepare all necessary data files
    
    Args:
        data_dir: Directory containing all data files
        
    Returns:
        Tuple of (tracking_df, plays_df, player_play_df)
    """
    print("Loading plays and player play data...")
    plays_df = pd.read_csv(os.path.join(data_dir, 'plays.csv'))
    player_play_df = pd.read_csv(os.path.join(data_dir, 'player_play.csv'))
    games_df = pd.read_csv(os.path.join(data_dir, 'games.csv'))
    
    print("Processing tracking data...")
    tracking_df = process_tracking_files(
        data_dir,
        plays_df,
        player_play_df,
        games_df,
        # [1],
    )
    
    # Generate summary statistics
    motion_stats = get_motion_summary(tracking_df, player_play_df)
    
    return tracking_df, plays_df, player_play_df, motion_stats

def plot_influence_map(analyzer, results, tracking_df, i, verbose=False):
    # Find play & grab frames
    example_play = list(results.values())[i]
    game_id, play_id = list(results.keys())[i]
    pre_motion_frameId = example_play['pre_motion_frameId']
    pre_snap_frameId = example_play['pre_snap_frameId']
    possession_team = example_play['possessionTeam']
    # Pre-motion player positions (offense & defense)
    pre_motion_frame_data = tracking_df[
        (tracking_df['gameId'] == game_id) & 
        (tracking_df['playId'] == play_id) & 
        (tracking_df['frameId'] == pre_motion_frameId)
    ]
    # Post-most player positions (offense & defense)
    pst_motion_frame_data = tracking_df[
        (tracking_df['gameId'] == game_id) & 
        (tracking_df['playId'] == play_id) & 
        (tracking_df['frameId'] == pre_snap_frameId)
    ]
    # Plot pre-motion influence map
    plot_influence_surface(
        analyzer.X, 
        analyzer.Y, 
        example_play['pre_motion'],
        f"{game_id}, {play_id}, {possession_team} - pre-motion",
        tracking_frame=pre_motion_frame_data,
        possession_team=possession_team
    )
    # Plot post-motion influence map
    plot_influence_surface(
        analyzer.X, 
        analyzer.Y, 
        example_play['pre_snap'],
        f"{game_id}, {play_id}, {possession_team} - post-motion",
        tracking_frame=pst_motion_frame_data,
        possession_team=possession_team
    )
    # Plot post minus pre-motion influence change
    plot_influence_surface(
        analyzer.X, 
        analyzer.Y, 
        example_play['change'],
        f"{game_id}, {play_id}, {possession_team} - change",
        tracking_frame=None,
        possession_team=possession_team
    )
    # If required, print results
    if verbose:
        x1 = example_play['pre_motion']
        x2 = example_play['pre_snap']
        print(f"pre-motion balance: {np.sum(x1[x1>0]) + np.sum(x1[x1<0]):0.2f}")
        print(f"post-motion balance: {np.sum(x2[x2>0]) + np.sum(x2[x2<0]):0.2f}")