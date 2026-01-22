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
from matplotlib.lines import Line2D


def plot_dfa_boxplots(df, data_points=True):
    """
    Boxplots of DFA_travel_distance across Set, Group, and Frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['Set', 'Group', 'Freq', 'DFA_travel_distance']

    data_points : bool
        If True, overlay jittered individual data points
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create combined x-axis category
    df = df.copy()
    df["Set_Group"] = (
        "Set " + df["Set"].astype(str) + " | " + df["Group"]
    )

    plt.figure(figsize=(18, 7))

    # Boxplot
    sns.boxplot(
        data=df,
        x="Set_Group",
        y="DFA_travel_distance",
        hue="Freq",
        dodge=True,
        width=0.7,
        showfliers=False
    )

    # Optional datapoints
    if data_points:
        sns.stripplot(
            data=df,
            x="Set_Group",
            y="DFA_travel_distance",
            hue="Freq",
            dodge=True,
            jitter=0.15,
            alpha=0.6,
            linewidth=0.5
        )

    # Fix duplicated legend (because of box + strip)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], title="Freq", frameon=True)

    plt.xlabel("Set | Group")
    plt.ylabel("DFA travel distance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_dfa_by_freq_sets_groups(df, data_points=True):
    """
    For each sampling frequency (10, 20, 40), plot DFA boxplots across
    Sets (1–5) with Group as hue.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['Set', 'Group', 'Freq', 'DFA_travel_distance']

    data_points : bool
        If True, overlay jittered datapoints
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    freqs = sorted(df["Freq"].unique())
    groups_order = ["Variable", "Non-variable", "Non-structured"]

    fig, axes = plt.subplots(
        1, len(freqs), figsize=(18, 6), sharey=True
    )

    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        df_f = df[df["Freq"] == freq]

        sns.boxplot(
            data=df_f,
            x="Set",
            y="DFA_travel_distance",
            hue="Group",
            hue_order=groups_order,
            dodge=True,
            width=0.7,
            showfliers=False,
            ax=ax
        )

        if data_points:
            sns.stripplot(
                data=df_f,
                x="Set",
                y="DFA_travel_distance",
                hue="Group",
                hue_order=groups_order,
                dodge=True,
                jitter=0.15,
                alpha=0.6,
                linewidth=0.5,
                ax=ax
            )

        ax.set_title(f"Sampling frequency = {freq} Hz")
        ax.set_xlabel("Set")
        ax.set_ylabel("DFA travel distance")

        # Remove duplicated legends per axis
        ax.legend_.remove()

    # Single legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:len(groups_order)],
        labels[:len(groups_order)],
        title="Group",
        loc="lower center",
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Writting\Publication tryouts\Journal of Sports Sciences\Review 1'
os.chdir(directory)
results = pd.read_excel(r'DFA in multiple sample frequencies (4-N4).xlsx')
print(results)

plot_dfa_by_freq_sets_groups(results, data_points=True)


# # Pivot to wide format
# df_wide = (
#     results
#     .pivot_table(
#         index=["ID", "Group"],
#         columns=["Set", "Freq"],
#         values="DFA_travel_distance"
#     )
# )
#
# # Flatten column names
# df_wide.columns = [
#     f"DFA_set{set_}_freq{freq}"
#     for set_, freq in df_wide.columns
# ]
#
# # Reset index to get ID and Group back as columns
# df_wide = df_wide.reset_index()
# df_wide.to_excel('DFA in multiple sample frequencies (4-N4) for spss.xlsx')