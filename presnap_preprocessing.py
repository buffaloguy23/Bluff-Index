#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:45:48 2024`
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import json

def load_data(root):
    # Load the static dataframes
    games_df = pd.read_csv(f'{root}/games.csv')
    plays_df = pd.read_csv(f'{root}/plays.csv')
    players_df = pd.read_csv(f'{root}/players.csv')
    player_play_df = pd.read_csv(f'{root}/player_play.csv')
    
    return games_df, plays_df, players_df, player_play_df

def load_tracking_data(root, week):
    return pd.read_csv(f'{root}/tracking_week_{week}.csv')

def normalize_coordinates(x, y, play_direction):
    if play_direction == 'right':
        return x, y
    else:  # play_direction == 'left'
        return 100 - x, 53.3 - y

def normalize_route(route, play_direction, start_x):
    normalized_route = []
    for x, y, vx, vy in route:
        norm_x, norm_y = normalize_coordinates(x, y, play_direction)
        norm_x -= start_x  # Shift to start at 0
        
        # Adjust velocities based on play direction
        if play_direction == 'left':
            vx, vy = -vx, -vy
        
        # Rotate coordinates and velocities 90 degrees clockwise
        rotated_x, rotated_y = norm_y, norm_x
        rotated_vx, rotated_vy = vy, vx
        
        normalized_route.append((rotated_x, rotated_y, rotated_vx, rotated_vy))
    
    return normalized_route

def extract_presnap_motion(tracking_data):
    routes = []
    
    # Filter for pre-snap motion players
    mask = (tracking_data['frameType'] == 'BEFORE_SNAP') & (tracking_data['motionSinceLineset'] == True)
    motion_players = tracking_data[mask]
    
    if len(motion_players) == 0:
        return [], False
    
    play_direction = tracking_data['playDirection'].iloc[0]
    start_x, _ = normalize_coordinates(tracking_data['absoluteYardlineNumber'].iloc[0], 0, play_direction)
    
    for player_id in motion_players['nflId'].unique():
        player_data = motion_players[motion_players['nflId'] == player_id].sort_values('frameId').reset_index(drop=True)
        
        # Get motion coordinates
        route = list(zip(player_data['x'], player_data['y'], 
                        player_data['s'] * np.cos(np.radians(player_data['dir'])), 
                        player_data['s'] * np.sin(np.radians(player_data['dir']))))
        normalized_route = normalize_route(route, play_direction, start_x)
        
        routes.append(normalized_route)
    
    return routes, True

def process_play(play_id, game_id, games_df, plays_df, player_play_df, tracking_df):
    # Merge all datasets
    game_data = games_df[games_df['gameId'] == game_id]
    play_data = plays_df[(plays_df['gameId'] == game_id) & (plays_df['playId'] == play_id)]
    player_play_data = player_play_df[(player_play_df['gameId'] == game_id) & (player_play_df['playId'] == play_id)]
    tracking_data = tracking_df[(tracking_df['gameId'] == game_id) & (tracking_df['playId'] == play_id)]

    # Merge all data into tracking_data
    tracking_data = tracking_data.merge(game_data, on='gameId', how='left')
    tracking_data = tracking_data.merge(play_data, on=['gameId', 'playId'], how='left')
    tracking_data = tracking_data.merge(player_play_data, on=['gameId', 'playId', 'nflId'], how='left')

    # Extract normalized routes and motion information
    routes, presnap_motion = extract_presnap_motion(tracking_data)
    
    # Calculate time remaining
    game_clock = tracking_data['gameClock'].iloc[0]
    total_seconds = (4 - tracking_data['quarter'].iloc[0]) * 900 + int(game_clock.split(':')[0]) * 60 + int(game_clock.split(':')[1])
    
    # Calculate score difference
    possession_team = tracking_data['possessionTeam'].iloc[0]
    home_team = tracking_data['homeTeamAbbr'].iloc[0]
    pre_snap_home = tracking_data['preSnapHomeScore'].iloc[0]
    pre_snap_visitor = tracking_data['preSnapVisitorScore'].iloc[0]
    
    if possession_team == home_team:
        score_difference = pre_snap_home - pre_snap_visitor
    else:
        score_difference = pre_snap_visitor - pre_snap_home
    
    # Prepare play data in the specified format
    formatted_play_data = {
        'routes': routes,
        'presnap_motion': presnap_motion,
        'metadata': {
            'season': tracking_data['season'].iloc[0],
            'week': tracking_data['week'].iloc[0],
            'down': tracking_data['down'].iloc[0],
            'distance': tracking_data['yardsToGo'].iloc[0],
            'field_position': tracking_data['absoluteYardlineNumber'].iloc[0],
            'time_remaining': total_seconds,
            'score_difference': score_difference,
            'num_receivers': len(routes),
            'playId': play_id,
            'gameId': game_id,
            'pass_result': tracking_data['passResult'].iloc[0],
            'offenseFormation': tracking_data['offenseFormation'].iloc[0],
            'receiver_alignment': tracking_data['receiverAlignment'].iloc[0],
            'pff_passCoverage': tracking_data['pff_passCoverage'].iloc[0],
            'pff_manZone': tracking_data['pff_manZone'].iloc[0],
            'possessionTeam': tracking_data['possessionTeam'].iloc[0],
        }
    }
        
    return formatted_play_data

def serialize_play_data(play_data):
    def serialize(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(play_data, default=serialize)

def main(stop_week):
    load_root = '/coding_projects/NFL Big Data Bowl 2025/raw_data'
    save_root = '/coding_projects/NFL Big Data Bowl 2025/curated_data'
    games_df, plays_df, players_df, player_play_df = load_data(load_root)
    play_data = []
    
    for week in tqdm(range(1, stop_week)):
        tracking_df = load_tracking_data(load_root, week)
        
        for _, game in games_df[games_df['week'] == week].iterrows():
            game_plays = plays_df[plays_df['gameId'] == game['gameId']]
            
            for _, play in game_plays.iterrows():
                processed_play = process_play(play['playId'], game['gameId'], games_df, plays_df, player_play_df, tracking_df)
                play_data.append(processed_play)
        
        # Save the processed data
        with open(f'{save_root}/processed_presnap_motion_data.json', 'w') as f:
            f.write(serialize_play_data(play_data))
        print("Data processing complete. Processed data saved to 'processed_presnap_motion_data.json'")

if __name__ == "__main__":
    main(10)