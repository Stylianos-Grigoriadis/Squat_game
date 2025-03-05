import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob


pd.set_option("display.max_rows", None)


directory_path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Test to fix target width and length'

os.chdir(directory_path)

parisPink_data = pd.read_csv('parisPink.txt')
parisPink_targets = pd.read_excel('parisPink.xlsx')

parisStatic_data = pd.read_csv('parisStatic.txt')
parisStatic_targets = pd.read_excel('parisStatic.xlsx')

parisWhite_data = pd.read_csv('parisWhite.txt')
parisWhite_targets = pd.read_excel('parisWhite.xlsx')



# Extract Excel file and convert them into real data
target_signal_x_pink, target_signal_y_pink = lbs.convert_excel_to_screen_size_targets(parisPink_targets, x_screen_size=1920-2*128, y_screen_size=1080-2*128)
target_signal_x_static, target_signal_y_static = lbs.convert_excel_to_screen_size_targets(parisStatic_targets, x_screen_size=1920-2*128, y_screen_size=1080-2*128)
target_signal_x_white, target_signal_y_white = lbs.convert_excel_to_screen_size_targets(parisWhite_targets, x_screen_size=1920-2*128, y_screen_size=1080-2*128)


target_signal_x_pink = target_signal_x_pink+128
target_signal_y_pink = target_signal_y_pink+128
target_signal_x_static = target_signal_x_static+128
target_signal_y_static = target_signal_y_static+128
target_signal_x_white = target_signal_x_white+128
target_signal_y_white = target_signal_y_white+128

target_signal_x_pink = np.round(target_signal_x_pink, decimals=0)
target_signal_y_pink = np.round(target_signal_y_pink, decimals=0)
target_signal_x_static = np.round(target_signal_x_static, decimals=0)
target_signal_y_static = np.round(target_signal_y_static, decimals=0)
target_signal_x_white = np.round(target_signal_x_white, decimals=0)
target_signal_y_white = np.round(target_signal_y_white, decimals=0)

# target_signal_x_pink = lbs.convert_target_position_X_if_old_data(target_signal_x_pink)
# target_signal_y_pink = lbs.convert_target_position_Y_if_old_data(target_signal_y_pink)
# target_signal_x_static = lbs.convert_target_position_X_if_old_data(target_signal_x_static)
# target_signal_y_static = lbs.convert_target_position_Y_if_old_data(target_signal_y_static)
# target_signal_x_white = lbs.convert_target_position_X_if_old_data(target_signal_x_white)
# target_signal_y_white = lbs.convert_target_position_Y_if_old_data(target_signal_y_white)


target_signal_x_pink = target_signal_x_pink[:30]
target_signal_y_pink = target_signal_y_pink[:30]
target_signal_x_static = target_signal_x_static[:30]
target_signal_y_static = target_signal_y_static[:30]
target_signal_x_white = target_signal_x_white[:30]
target_signal_y_white = target_signal_y_white[:30]




parisPink_data = lbs.values_during_game(parisPink_data)
parisStatic_data = lbs.values_during_game(parisStatic_data)
parisWhite_data = lbs.values_during_game(parisWhite_data)

_, indices_before_change_pink = lbs.find_the_last_moment_before_target_change_position(parisPink_data['target_pos_x'])
_, indices_before_change_static = lbs.find_the_last_moment_before_target_change_position(parisStatic_data['target_pos_x'])
_, indices_before_change_white= lbs.find_the_last_moment_before_target_change_position(parisWhite_data['target_pos_x'])


indices_before_change_pink = indices_before_change_pink[1:]
indices_before_change_static = indices_before_change_static[1:]
indices_before_change_white = indices_before_change_white[1:]

target_generated_paris_pink_X = []
target_generated_paris_pink_Y = []
for index in indices_before_change_pink:
    target_generated_paris_pink_X.append(parisPink_data['target_pos_x'][index])
    target_generated_paris_pink_Y.append(parisPink_data['target_pos_y'][index])

target_generated_paris_static_X = []
target_generated_paris_static_Y = []
for index in indices_before_change_static:
    target_generated_paris_static_X.append(parisStatic_data['target_pos_x'][index])
    target_generated_paris_static_Y.append(parisStatic_data['target_pos_y'][index])

target_generated_paris_white_X = []
target_generated_paris_white_Y = []
for index in indices_before_change_white:
    target_generated_paris_white_X.append(parisWhite_data['target_pos_x'][index])
    target_generated_paris_white_Y.append(parisWhite_data['target_pos_y'][index])

for i in range(len(target_generated_paris_pink_X)):
    print(f'{target_generated_paris_pink_X[i]}            {target_signal_x_pink[i]}')

plt.scatter(target_generated_paris_pink_X, target_generated_paris_pink_Y, label='Generated')
plt.scatter(target_signal_x_pink, target_signal_y_pink, label='Excel')
plt.legend()
plt.show()

plt.scatter(target_generated_paris_static_X, target_generated_paris_static_Y, label='Generated')
plt.scatter(target_signal_x_static, target_signal_y_static, label='Excel')
plt.legend()
plt.show()

plt.scatter(target_generated_paris_white_X, target_generated_paris_white_Y, label='Generated')
plt.scatter(target_signal_x_white, target_signal_y_white, label='Excel')
plt.legend()
plt.show()

plt.plot(target_generated_paris_pink_X, label='Generated')
plt.plot(target_signal_x_pink, label='Excel')
plt.legend()
plt.show()


plt.plot(target_generated_paris_pink_Y, label='Generated')
plt.plot(target_signal_y_pink, label='Excel')
plt.legend()
plt.show()





# Extract the



# Create a list with 5 sublists which contain 30 dataframes, each dataframe contains all data of each target
# list_with_all_df_separated_by_set = lbs.return_the_df_of_each_target_separated_by_set(data)
# print(list_with_all_df_separated_by_set[3])

# # Comparison of Excel signal and the generated signal
# target_pos_x_list = []
# target_pos_y_list = []
#
# for list in list_with_all_df_separated_by_set:
#     for df in list:
#         target_pos_x_list.append(df['target_pos_x'][0])
#         target_pos_y_list.append(df['target_pos_y'][0])
# for i in range(len(target_pos_x_list)):
#     if target_pos_x_list[i] < 128:
#         target_pos_x_list[i]= 128
#     if target_pos_x_list[i] > 1980 - 128:
#         target_pos_x_list[i] = 1980 - 128
#
# for i in range(len(target_pos_y_list)):
#     if target_pos_y_list[i] < 128:
#         target_pos_y_list[i]= 128
#     if target_pos_y_list[i] > 1080 - 128:
#         target_pos_y_list[i] = 1080 - 128
#
#
# plt.scatter(target_pos_x_list, target_pos_y_list, label='Generated')
# plt.scatter(target_signal_x, target_signal_y, label='Excel')
# plt.legend()
# plt.show()
#
# plt.plot(target_pos_x_list, label='Generated')
# plt.plot(target_signal_x, label='Excel')
# plt.legend()
# plt.show()





# Create a simple graph with all the columns you need to plot
# lbs.simple_graph(list_with_all_df_separated_by_set, 'pitch', 'yaw', 'roll')

# Create a scatter with a slider for visualization of target position vs player position
# lbs.graph_creation_target_vs_player(list_with_all_df_separated_by_set)
# lbs.graph_creation_target_vs_player_with_data(list_with_all_df_separated_by_set)
#
#
#
# spatial_error = lbs.spatial_error_best_window(list_with_all_df_separated_by_set, plot=True, time_window=300)

