import lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

pd.set_option("display.max_rows", None)

directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 4\Data'
os.chdir(directory)
targets = pd.read_excel(r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 4\Signals\Participant 4 pink 150 targets.xlsx')
# print(targets)
# print(targets.columns)
data = pd.read_csv(r'Participant 4 pink 150 targets.txt')
start_value = 1
increment = 55
num_values = 150

# Generate the signal
time = np.arange(start_value, start_value + increment * num_values, increment)

X_coordinates = targets['X coordinates'].to_numpy()
X_coordinates = (X_coordinates - np.min(X_coordinates)) / (np.max(X_coordinates) - np.min(X_coordinates)) * 1920

target_pos_x, target_pos_y, player_pos_x, player_pos_y = lbs.values_during_game(data)

target_pos_x_each_target, target_pos_y_each_target, player_pos_x_each_target, player_pos_y_each_target = lbs.separate_data_for_each_target(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
sublists_of_sets, index_for_set_seperation = lbs.find_the_last_moment_before_target_change_position(target_pos_x)
spatial_error_each_target = lbs.calculate_spatial_error_for_each_target(target_pos_x, target_pos_y, player_pos_x, player_pos_y, 15)


plt.plot(spatial_error_each_target, label='Spatial Error')

plt.axvline(x=0, lw=0.5, linestyle='--', c='k')
for x in index_for_set_seperation:
    plt.axvline(x=x, lw=0.5, linestyle='--', c='k')
plt.axvline(x=len(spatial_error_each_target), lw=0.5, linestyle='--', c='k', label='Set separator')

plt.legend()
plt.show()
# target_pos_x_last, target_pos_y_last, player_pos_x_last, player_pos_y_last = lbs.return_the_values_before_target_change(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
#
# time_of_change = lbs.find_the_last_moment_before_target_change_position(target_pos_y)
# time_of_change = [item for sublist in time_of_change for item in sublist]
# target_pos_x_each_target, target_pos_y_each_target, player_pos_x_each_target, player_pos_y_each_target = lbs.separate_data_for_each_target(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
# target_pos_y_each_target = [item for sublist in target_pos_y_each_target for item in sublist]
# print(target_pos_y_each_target)
# print(len(time_of_change))
# print(len(X_coordinates[:146]))
# print(len(target_pos_x_last))
# print(target_pos_x_last[29])
# print(target_pos_x_last[30])
# print(target_pos_x_last[31])

# plt.plot(target_pos_y, label='target_pos_x')
# plt.plot(target_pos_y_each_target, label='target_pos_y_each_target')
# plt.scatter(time_of_change, target_pos_y_last, c='red')
#
# for x in time_of_change:
#     plt.axvline(x=x, lw=0.5, linestyle='--', c='k')
# plt.legend()
# plt.show()
# spatial_error_all = lbs.spatial_error_calculation(target_pos_x, target_pos_y, player_pos_x, player_pos_y)
#


# spatial_error_last = lbs.spatial_error_calculation(target_pos_x_last, target_pos_y_last, player_pos_x_last, player_pos_y_last)

# plt.title('All targets and players positions during game')
# plt.scatter(target_pos_x, target_pos_y, label='target')
# plt.scatter(player_pos_x, player_pos_y, label='player')
# plt.legend()
# plt.show()
#
# plt.title('Only the last moment before target changes')
# plt.scatter(target_pos_x_last, target_pos_y_last, label='target')
# plt.scatter(player_pos_x_last, player_pos_y_last, label='player')
# plt.legend()
# plt.show()
#
# plt.plot(spatial_error_all, label='all spatial error')
# plt.legend()
# plt.show()
#
# plt.plot(spatial_error_last, label='spatial error only the moment before change')
# plt.legend()
# plt.show()



