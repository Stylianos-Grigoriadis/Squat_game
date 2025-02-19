import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
import colorednoise as cn
from scipy.optimize import direct

import Lib_squats as lbs
import math
from scipy.constants import g


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Data collection\Data\pink1'
os.chdir(directory)
data = pd.read_csv('pink1.csv')


