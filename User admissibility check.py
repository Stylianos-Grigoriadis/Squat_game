import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns


pd.set_option("display.max_rows", None)

participants_before_change = ['pink1', 'pink10', 'pink11', 'pink12', 'pink13', 'pink14','pink15', 'pink2', 'pink3', 'pink4', 'pink5', 'pink6', 'pink7', 'pink8', 'pink9', 'static1', 'static10', 'static11', 'static12', 'static13', 'static2', 'static3', 'static4', 'static5', 'static6', 'static7', 'static8', 'static9', 'white1', 'white10', 'white11', 'white12', 'white13', 'white14', 'white2', 'white3', 'white4', 'white5', 'white6', 'white7', 'white8', 'white9']

directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Data'
files = glob.glob(os.path.join(directory_path, "*"))

list_ID = []

for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    print(ID)
    if ID in participants_before_change:
        old_data = True
        print('Data taken before the fix')
    else:
        old_data = False
        print('Data taken after the fix')
    print(f'old data = {old_data}')

    if 'static' not in ID:
        targets = pd.read_excel(rf'{ID}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
        if 'pink' in ID:
            list_ID.append('pink')
        else:
            list_ID.append('white')
    else:
        targets = pd.read_excel(rf'{ID[:6]}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
        list_ID.append('static')

    # Extract Excel file and convert them into real data
    target_signal_x, target_signal_y = lbs.convert_excel_to_screen_size_targets(targets, old_data)

    # Extract the
    data = lbs.values_during_game(data)


    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data, old_data)


    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')


    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)


    # Calculate and print the spatial error for each target
    spatial_error, list_time_stamp_of_min_spatial_error_separated_by_set = lbs.spatial_error_best_window(
        list_with_all_df_separated_by_set, plot=False, time_window=500)
