import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from fathon import fathonUtils as fu
import fathon
import lib_squats as lbsq


path = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Pilot Study 1\Stylianos 11.5.2024'
squat = pd.read_csv(path+r'\Squat.csv')
on_table = pd.read_csv(path+r'\Stable on table.csv')
squat_0_degree = pd.read_csv(path+r'\Squat_0_degree.csv')
squat_180_degree = pd.read_csv(path+r'\Squat_180_degree.csv')

squat = lbsq.convert_KIVNENT_IMU_to_readable_file(squat)
on_table = lbsq.convert_KIVNENT_IMU_to_readable_file(on_table)
squat_0_degree = lbsq.convert_KIVNENT_IMU_to_readable_file(squat_0_degree)
squat_180_degree = lbsq.convert_KIVNENT_IMU_to_readable_file(squat_180_degree)

yaw = squat_180_degree['Yaw'].to_numpy()
pitch = squat_180_degree['Pitch'].to_numpy()
roll = squat_180_degree['Roll'].to_numpy()

yaw = yaw
pitch = pitch
roll = roll

print(squat)
print(on_table)
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(squat['Yaw'], color='red', label='Yaw')
# axs[0].plot(squat['Pitch'], color='blue', label='Pitch')
# axs[0].plot(squat['Roll'], color='green', label='Roll')
# axs[0].set_title('Squat')
# axs[0].legend()
#
#
# axs[1].plot(on_table['Yaw'], color='red', label='Yaw')
# axs[1].plot(on_table['Pitch'], color='blue', label='Pitch')
# axs[1].plot(on_table['Roll'], color='green', label='Roll')
# axs[1].set_title('On table')
# axs[1].legend()
#
# plt.tight_layout()
# plt.show()

fig, axs = plt.subplots(3, 1)
axs[0].plot(squat['Yaw'], color='red', label='squat')
axs[0].plot(squat_0_degree['Yaw'], color='blue', label='squat_0_degree')
axs[0].plot(yaw, color='green', label='squat_180_degree')
axs[0].set_title('Yaw')
axs[0].legend()

axs[1].plot(squat['Pitch'], color='red', label='squat')
axs[1].plot(squat_0_degree['Pitch'], color='blue', label='squat_0_degree')
axs[1].plot(pitch, color='green', label='squat_180_degree')
axs[1].set_title('Pitch')
axs[1].legend()

axs[2].plot(squat['Roll'], color='red', label='squat')
axs[2].plot(squat_0_degree['Roll'], color='blue', label='squat_0_degree')
axs[2].plot(roll, color='green', label='squat_180_degree')
axs[2].set_title('Roll')
axs[2].legend()

plt.tight_layout()
plt.show()


