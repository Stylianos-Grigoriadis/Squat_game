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

df_long['ID'] = pd.Categorical(df_long['ID'], categories=["static", "pink", "white"], ordered=True)

# Custom color palette
custom_palette = {
    "static": "#4F4F4F",    # Dark gray (first)
    "pink": "#FFC0CB",      # Soft pink (second)
    "white": "#D3D3D3"      # Light gray (third)
}

# Create the boxplot with the correct order
plt.figure(figsize=(12, 6))
sns.boxplot(x='Set', y='Average Spatial Error', hue='ID', data=df_long, palette=custom_palette, showfliers=False)

# Customize plot
plt.title('Comparison of Spatial Error Across Sets and ID Groups')
plt.xlabel('Set')
plt.ylabel('Average Spatial Error')
plt.legend(title='ID')

# Show plot
plt.show()

# Plot Slopes
df_melted_Slope = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression Slope',
    'Segmented Regression Slope before',
    'Segmented Regression Slope after'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_Slope, showfliers=False)
plt.title('Slope by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()



# Plot intercepts
df_melted_intercept = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression Intercept',
    'Segmented Regression Intercept before',
    'Segmented Regression Intercept after'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_intercept, showfliers=False)
plt.title('Intercept by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()



# Plot RMSE
df_melted_rmse = df.melt(id_vars=['ID'], value_vars=[
    'Simple Regression RMSE',
    'Segmented Regression RMSE'
], var_name='Regression Type', value_name='Value')

# Set up the plotting style (optional)
sns.set(style="whitegrid")

# Create the boxplot with 3 major columns (regression types) and hues (ID)
plt.figure(figsize=(14, 8))
sns.boxplot(x='Regression Type', y='Value', hue='ID', data=df_melted_rmse, showfliers=False)
plt.title('RMSE by ID')
plt.xticks(rotation=35, ha='right')  # Rotate the x-axis labels for better readability
plt.legend(title='ID')
plt.gca().set_xlabel('')  # Remove the "Regression Type" label without leaving space
plt.tight_layout()
plt.show()


# Plot Average Spatial Error
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightgreen', 'lightcoral']  # Adjust colors as needed
sns.boxplot(data=df, x='ID', y='Average Spatial Error all sets', hue='ID', palette=colors, legend=False)
plt.title('Average Spatial Error by ID')
plt.xlabel('ID')
plt.ylabel('Average Spatial Error')
plt.show()



# Plot Sd Spatial Error
plt.figure(figsize=(10, 6))
colors = ['lightblue', 'lightgreen', 'lightcoral']  # Adjust colors as needed
sns.boxplot(data=df, x='ID', y='Sd Spatial Error all sets', palette=colors)
plt.title('Sd Spatial Error by ID')
plt.xlabel('ID')
plt.ylabel('Sd Spatial Error')
plt.show()



