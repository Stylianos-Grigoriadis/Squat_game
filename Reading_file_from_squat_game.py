import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 2'
os.chdir(directory)


signal = pd.read_csv(r'Stylianos_1_7_2024_12.04.txt')
signal['target_pos_x'] = signal['target_pos_x'].replace(' None', np.nan).astype(float)
signal['target_pos_y'] = signal['target_pos_y'].replace(' None', np.nan).astype(float)

print(signal)
# signal.to_excel(rf'{directory}\Stylianos_1_7_2024_12.04.xlsx')
print(signal.columns)
print(signal['timestamp'])
print(signal['player_pos_x'])
print(signal['target_pos_x'])
print('types')
print()

yaw = signal['yaw'][1285:2502].to_numpy()
print(yaw)
min_angle = float(signal['min_angle'][1285])
print(min_angle)

max_angle = float(signal['max_angle '][1285])
print(max_angle)
print(type(yaw), type(max_angle), type(min_angle))
player_position_x_from_game = signal['player_pos_x'][1285:2502].to_numpy()
player_position_x_from_game = player_position_x_from_game/100
player_position_x = ((yaw - max_angle)/(min_angle - max_angle))*100
plt.plot(yaw, label='yaw')
plt.plot(player_position_x, label='player_position_x_calculated')
plt.plot(player_position_x_from_game, label='player_position_x_from_game')
plt.legend()
plt.show()


plt.scatter(signal['player_pos_x'][1284:2502], signal['player_pos_y'][1284:2502], label='player_pos')
plt.scatter(signal['target_pos_x'][1284:2502], signal['target_pos_y'][1284:2502], label='target_pos')
plt.legend()
plt.show()


