import Lib_squats as lbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import Slider
import glob
import seaborn as sns
import pwlf
import piecewise_regression
from scipy.stats import linregress
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

spatial_error = True
if spatial_error:

    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Results'
    os.chdir(directory)
    df = pd.read_excel(r'Results after exclusion of outliers.xlsx')

    print(df.columns)



    df_long = df.melt(id_vars=['ID'],
                      value_vars=['Average Spatial error set 1', 'Average Spatial error set 2',
                                  'Average Spatial error set 3', 'Average Spatial error set 4',
                                  'Average Spatial error set 5'],
                      var_name='Set', value_name='Average Spatial Error')  # Ensure correct name



    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('Average Spatial error set ', 'Set ')

    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Repeated", "Pink Noise", "White Noise"], ordered=True)

    # Custom color palette
    custom_palette = {
        "Repeated": "#4F4F4F",    # Dark gray (first)
        "Pink Noise": "#FFC0CB",      # Soft pink (second)
        "White Noise": "#D3D3D3"      # Light gray (third)
    }

    # Create the boxplot with the correct order
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='Average Spatial Error', hue='ID', data=df_long, palette=custom_palette, showfliers=False)

    # Customize plot
    plt.title('Spatial Error Across Sets and Groups')
    plt.ylabel('Average Spatial Error')
    plt.xlabel('')
    plt.legend()
    plt.ylim(15,400)

    # Show plot
    plt.show()


else:
    directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Results'
    os.chdir(directory)
    df_learning = pd.read_excel(r'Results of Non-linear analysis.xlsx')

    print(df_learning.columns)


    df_long = df_learning.melt(id_vars=['ID'],
                               value_vars=['SaEn_travel_distance_set_1', 'SaEn_travel_distance_set_2',
                                           'SaEn_travel_distance_set_3', 'SaEn_travel_distance_set_4',
                                           'SaEn_travel_distance_set_5'],
                               var_name='Set', value_name='Sample Entropy')  # Ensure correct name

    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('Sample Entropy ', 'Set ')
    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Repeated", "Pink Noise", "White Noise"], ordered=True)

    # Create boxplot
    custom_palette = {
        "Repeated": "#4F4F4F",  # Dark gray (first)
        "Pink Noise": "#FFC0CB",  # Soft pink (second)
        "White Noise": "#D3D3D3"  # Light gray (third)
    }
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='Sample Entropy', hue='ID', data=df_long, palette=custom_palette, showfliers=False)

    custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']
    plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)

    # Customize plot
    plt.title('Sample Entropy of TD Across Sets Between Groups')
    plt.xlabel('')
    plt.ylabel('Sample Entropy')
    plt.legend()
    plt.ylim(0.25, 1.7)


    # Show plot
    plt.show()

    df_long = df_learning.melt(id_vars=['ID'],
                               value_vars=['DFA_travel_distance_set_1', 'DFA_travel_distance_set_2',
                                           'DFA_travel_distance_set_3', 'DFA_travel_distance_set_4',
                                           'DFA_travel_distance_set_5'],
                               var_name='Set', value_name='DFA')  # Ensure correct name

    # Rename set names for readability
    df_long['Set'] = df_long['Set'].str.replace('DFA ', 'Set ')
    df_long['ID'] = pd.Categorical(df_long['ID'], categories=["Repeated", "Pink Noise", "White Noise"], ordered=True)

    # Create boxplot
    custom_palette = {
        "Repeated": "#4F4F4F",  # Dark gray (first)
        "Pink Noise": "#FFC0CB",  # Soft pink (second)
        "White Noise": "#D3D3D3"  # Light gray (third)
    }
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Set', y='DFA', hue='ID', data=df_long, palette=custom_palette,
                showfliers=False)  # Use correct column name
    custom_labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']
    plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)

    # Customize plot
    plt.title('Detrended Fluctuation Analysis of TD Across Sets and Between Groups')
    plt.xlabel('')
    plt.ylabel('α exponent')
    plt.legend()
    plt.ylim(0.43, 1.1)

    # Show plot
    plt.show()



