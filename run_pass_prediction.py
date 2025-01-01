#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:32:00 2024
"""
# Standard packages
import pandas as pd
import numpy as np
from tqdm import tqdm
# Plotting packages
import matplotlib.pyplot as plt
# ML packages
from sklearn.ensemble import RandomForestClassifier
# User-defined scripts
import player_influence
import presnap_motion
import functions

# %%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## PROCESS INFLUNCE DATA

# Load & process data
data_dir = "/coding_projects/NFL Big Data Bowl 2025/raw_data"
tracking_df, plays_df, player_play_df, motion_stats = player_influence.load_and_prepare_data(data_dir)
plays_df['id'] = plays_df['gameId'].astype(str) + '_' + plays_df['playId'].astype(str)

# Initialize analyzer
analyzer = player_influence.MotionAnalyzer(tracking_df, plays_df, player_play_df)

# Analyze all motion plays
results = analyzer.analyze_all_motion_plays()

# Plot example play
player_influence.plot_influence_map(analyzer, results, tracking_df, i=1, verbose=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## UNPACK INFLUENCE RESULTS
influence = []
for i in range(len(results)):
    game_id, play_id = list(results.keys())[i]
    example_play = list(results.values())[i]
    x1 = example_play['pre_motion']
    x2 = example_play['pre_snap']
    influence.append([
        game_id,
        play_id,
        np.sum(x1[x1>0]) + np.sum(x1[x1<0]),
        np.sum(x2[x2>0]) + np.sum(x2[x2<0]),
        ])
influence_df = pd.DataFrame(influence, columns=['gameId','playId','pre_motion_influence','post_motion_influence'])
influence_df['id'] = influence_df['gameId'].astype(str) + '_' + influence_df['playId'].astype(str)
influence_df['influence_change'] = (influence_df['post_motion_influence'] - influence_df['pre_motion_influence']) / influence_df['pre_motion_influence']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## LOAD GAME DATA & ADD TO PLAYS_DF

# Load & add to main DF
game_data = pd.read_csv('/coding_projects/NFL Big Data Bowl 2025/raw_data/games.csv')
plays_df = plays_df.merge(how='left', on='gameId', right=game_data[['gameId','season','week','gameDate','homeTeamAbbr', 'visitorTeamAbbr']])

# Add derived features
game_clock = plays_df['gameClock']
plays_df['time_remaining'] = (4 - plays_df['quarter'].iloc[0]) * 900 + \
    plays_df['gameClock'].apply(lambda x: str(x).split(':')[0]).astype(int) * 60 + \
    plays_df['gameClock'].apply(lambda x: str(x).split(':')[1]).astype(int)
plays_df['home_team_flag'] = plays_df['possessionTeam'] == plays_df['homeTeamAbbr']
plays_df['home_mov'] = plays_df['preSnapHomeScore'] - plays_df['preSnapVisitorScore']

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## GET PRESNAP MOTION LABELS FROM UNSUPERIZED AUTOENCODER

# Get offensive pre-snap encoding
off_ps_encoding = presnap_motion.run(
    presnap_filename='processed_presnap_motion_data.json', 
    plot_flag=True,
    seed=13, #17, #42
    )
off_ps_encoding['id'] = off_ps_encoding['gameId'].astype(str) + '_' + off_ps_encoding['playId'].astype(str)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## ADD PRE-SNAP INFORMATION TO MAIN DF

# Add pre-snap motion categorization to the main df
plays_df_aug = plays_df.merge(how='left', on=['gameId','playId'], right=off_ps_encoding[['gameId','playId','pc1','pc2','pc3','presnap_label']])

# Add line-of-scrimmage (LoS) battle results
plays_df_aug = plays_df_aug.merge(how='left', on=['gameId','playId'], right=influence_df.loc[np.isfinite(influence_df['influence_change']), ['gameId','playId','influence_change']])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## RUN VS PASS PREDICTIONS

# Set # of bins for calibration assessment
n_bins = 10

# Set input (x) & output (y) features
y_features = ['isDropback']
x_features = ['quarter', 'down', 'yardsToGo', 'time_remaining', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 
              'offenseFormation', 'receiverAlignment', 'home_team_flag', 'home_mov']
x_features_motion = ['quarter', 'down', 'yardsToGo', 'time_remaining', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 
              'offenseFormation', 'receiverAlignment', 'home_team_flag', 'home_mov',
              'presnap_label', 'influence_change']

# Label encode
categorical_columns = ['offenseFormation', 'receiverAlignment']
plays_df_aug, encoders = functions.encode_categorical_columns(plays_df_aug, categorical_columns)

# Generate predictions one week at a time
y_te_save = pd.DataFrame()
for week in tqdm(plays_df_aug['week'].sort_values().unique(), desc='Training probability model'):
    # Set train data
    tr_mask = plays_df_aug['week'] != week
    # X_tr = plays_df_aug.loc[tr_mask, x_features]
    X_tr = plays_df_aug.loc[tr_mask, x_features_motion]
    X_tr_motion = plays_df_aug.loc[tr_mask, x_features_motion]
    y_tr = plays_df_aug.loc[tr_mask, y_features].to_numpy().ravel()
    # Set test data
    te_mask = plays_df_aug['week'] == week
    # X_te = plays_df_aug.loc[te_mask, x_features]
    X_te = plays_df_aug.loc[te_mask, x_features_motion]
    X_te_motion = plays_df_aug.loc[te_mask, x_features_motion]
    y_te = plays_df_aug.loc[te_mask, y_features]
    
    # Spoof motion features
    X_tr[list(set(x_features_motion) - set(x_features))] = None
    X_te[list(set(x_features_motion) - set(x_features))] = None

    ## EXCLUDE MOTION
    # Train a Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    
    ## INCLUDE MOTION
    # Train a Random Forest Classifier
    rf_motion = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_motion.fit(X_tr_motion, y_tr)
    y_prob_motion = rf_motion.predict_proba(X_te_motion)[:, 1]
    
    ## COMPILE PREDICTIONS
    y_te['proba_pre'] = y_prob
    y_te['proba_post'] = y_prob_motion
    y_te_save = y_te[['proba_pre','proba_post']] if y_te_save.empty else pd.concat([y_te_save, y_te[['proba_pre','proba_post']]], axis=0)

# Add pre & post motion run-play probabilistic predictions to main DF
print(plays_df_aug.shape)
motion_eval_df = pd.concat([plays_df_aug, y_te_save], axis=1)
print(motion_eval_df.shape)
# Filter to only "test" points
motion_eval_df = motion_eval_df[motion_eval_df['proba_pre'].notnull()].reset_index(drop=True)
# Add post vs pre probability change feature
motion_eval_df['prob_change'] = motion_eval_df['proba_post'] - motion_eval_df['proba_pre']
# Add winProbabilityAdded feature
motion_eval_df['winProbabilityAdded'] = motion_eval_df['homeTeamWinProbabilityAdded']
mask = (motion_eval_df['possessionTeam']==motion_eval_df['visitorTeamAbbr'])
motion_eval_df.loc[mask,'winProbabilityAdded'] = motion_eval_df.loc[mask,'visitorTeamWinProbilityAdded']
# Create derived feature(s)
motion_eval_df['winProbabilityAddedx100'] = motion_eval_df['winProbabilityAdded'] * 100

# Show calibration accuracies
functions.plot_calibration_curve(motion_eval_df[y_features], motion_eval_df['proba_pre'], n_bins) # MOTION EXCLUDED
functions.plot_calibration_curve(motion_eval_df[y_features], motion_eval_df['proba_post'], n_bins) # MOTION INCLUDED
functions.plot_calibration_curve(motion_eval_df.loc[motion_eval_df['pc1'].notnull(),y_features], motion_eval_df.loc[motion_eval_df['pc1'].notnull(),'proba_post'], n_bins) # ONLY MOTION PLAYS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## COMPUTE CDF DISTANCE BETWEEN MOTION & NO-MOTION PLAYS
##      CONSTRAIN BY EP BUCKETS TO COMPARE APPLES-TO-APPLES

# Runtime parameters
plot_flag = False
bin_feature = 'prob_change'
target_feature = 'expectedPointsAdded'
play_type = True

# Bin data
n_bins = 20
n_bins_ep = 20
bin_edges = pd.cut(motion_eval_df.loc[motion_eval_df['pc1'].notnull(), bin_feature], n_bins, retbins=True)[1]
motion_eval_df['bin_prob'] = pd.cut(motion_eval_df[bin_feature], bins=bin_edges)
motion_eval_df['bin_midpoint'] = motion_eval_df['bin_prob'].apply(lambda x: (x.left + x.right) / 2)
motion_eval_df['bin_ep'] = pd.qcut(motion_eval_df['expectedPoints'], n_bins_ep, duplicates='drop')

# Loop thru bins
total = []
total_matrix = []
for bin_x in motion_eval_df['bin_prob'].sort_values().unique():
    for bin_y in motion_eval_df['bin_ep'].sort_values().unique():
        # Separate data into plays that are motion vs no-mtion in bin_x & bin_y
        no_motion_mask = (motion_eval_df['bin_ep']==bin_y) & (motion_eval_df['isDropback']==play_type) & (motion_eval_df['pc1'].isnull())
        motion_mask = (motion_eval_df['bin_prob']==bin_x) & (motion_eval_df['bin_ep']==bin_y) & (motion_eval_df['isDropback']==play_type) & (motion_eval_df['pc1'].notnull())
        data_no_motion = motion_eval_df.loc[no_motion_mask, target_feature]
        data_motion = motion_eval_df.loc[motion_mask, target_feature]
        
        # Compute the distance between the two CDF curves
        distances = functions.compute_cdf_distance(data_motion, data_no_motion)
        # Add results to lists
        total_matrix.append((bin_x, bin_y, distances['mean_distance']))
        total.append(distances['mean_distance'])
        # Plot if necessary
        if plot_flag:
            plt.figure()
            # For no-motion data
            counts_no_motion, bins_no_motion = np.histogram(data_no_motion, bins=50)
            cdf_no_motion = np.cumsum(counts_no_motion) / counts_no_motion.sum()
            plt.plot(bins_no_motion[1:], cdf_no_motion, label='no-motion')
            # For motion data
            counts_motion, bins_motion = np.histogram(data_motion, bins=50)
            cdf_motion = np.cumsum(counts_motion) / counts_motion.sum()
            plt.plot(bins_motion[1:], cdf_motion, label='motion-post')
            # Finish the plot
            plt.legend()
            plt.grid('minor')
            plt.xlabel('Expected Points Added')
            plt.ylabel('Cumulative Probability')
            plt.title(f"bin = {bin_x}, {bin_y}")
            plt.text(0.55, 0.05, 
                      f"Mean distance between curves: {distances['mean_distance']:.3f}"
                      f"\nMedian distance between curves: {distances['median_distance']:.3f}",
                      transform=plt.gca().transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))
            plt.show()
print(f"\nmean shift: {np.nanmean(total):0.2f} {target_feature} per play\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## QUANTIFY EPA-IMPROVEMENT FROM MOTION

# Put for-loop data into a DF
matrix_data = pd.DataFrame(total_matrix, columns=['bin_prob','bin_ep','distance'])

# Show correlation plot
mask_N = (motion_eval_df['pc1'].notnull()) & (motion_eval_df['isDropback']==play_type)
corr_df = matrix_data.groupby('bin_prob')['distance'].mean().reset_index()
corr_df['bin_midpoint'] = corr_df['bin_prob'].apply(lambda x: (x.left + x.right) / 2)
corr_df = corr_df.merge(
    how='left',
    on='bin_prob',
    right=motion_eval_df[mask_N].groupby('bin_prob')['id'].count().rename('count').reset_index()
    )

corr_df = corr_df[corr_df['count']>=.33 * mask_N.sum() / n_bins] # 50]
# functions.plot_bisector(corr_df['bin_midpoint'], corr_df['distance'])
functions.create_modern_correlation_plot(corr_df['bin_midpoint'], corr_df['distance'], title='', xlabel='Bluff Index', ylabel='EPA Improvement')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## MOTION-DISGUISE-INDEX BY TEAM

mask_antiN = (motion_eval_df['pc1'].isnull()) & (motion_eval_df['isDropback']==play_type)
team_mdi = motion_eval_df[mask_N].groupby('possessionTeam')['prob_change'].agg(['mean','median','count']).reset_index()\
    .sort_values(by='count', ascending=False).reset_index(drop=True)
team_mdi = team_mdi.merge(
    how='left',
    on='possessionTeam',
    right=motion_eval_df[mask_N].groupby('possessionTeam')['expectedPointsAdded'].agg(['mean','median','sum']),
    suffixes=('_dP','_epa')
    ).merge(
    how='left',
    on='possessionTeam',
    right=motion_eval_df[mask_antiN].groupby('possessionTeam')['expectedPointsAdded'].agg(['mean','median','sum','count']),
    suffixes=('','_epa_nomotion')
    ).rename(columns={'mean':'mean_epa_nomotion', 'median':'median_epa_nomotion'})
team_mdi['usage_rate'] = team_mdi['count'] / (team_mdi['count'] + team_mdi['count_epa_nomotion'])
team_mdi['+mean_epa'] = team_mdi['mean_epa'] - team_mdi['mean_epa_nomotion']
team_mdi['+median_epa'] = team_mdi['median_epa'] - team_mdi['median_epa_nomotion']

# Regular scatter plots
functions.plot_bisector(team_mdi['median_dP'], team_mdi['+median_epa'])

# Create and show plot with NFL logos
fig = functions.create_nfl_logo_correlation_plot(
    team_mdi['median_dP'], 
    team_mdi['+median_epa'],
    team_mdi['possessionTeam'], 
    '/coding_projects/NFL Big Data Bowl 2025/logos',
    img_type='.png',
    logo_display_size=(50, 50),
    title='Passing Plays with Motion',
    xlabel='Median Bluff Index',
    ylabel='Median EPA per Play',
)
plt.show()





 