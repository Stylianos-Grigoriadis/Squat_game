import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob


pd.set_option("display.max_rows", None)


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Data'
files = glob.glob(os.path.join(directory_path, "*"))
for file in files:
    os.chdir(file)
    ID = os.path.basename(file)
    print(ID)

    if not ID[:-1] == 'static':
        targets = pd.read_excel(rf'{ID}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
    else:
        targets = pd.read_excel(rf'{ID[:-1]}.xlsx')
        data = pd.read_csv(rf'{ID}.txt')
    # Extract Excel file and convert them into real data
    target_signal_x, target_signal_y = lbs.convert_excel_to_screen_size_targets(targets)

    # Extract the
    data = lbs.values_during_game(data)
    print(data.columns)


    # Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
    list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data)
    # print(list_with_all_df_separated_by_set[3])



    # Create a simple graph with all the columns you need to plot
    # lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

    # Create a scatter with a slider for visualization of target position vs player position
    # lbs.graph_creation_target_vs_player(list_with_all_df_separated_by_set)

    lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)



    # spatial_error = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=True, time_window=300)

