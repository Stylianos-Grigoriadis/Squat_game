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

def plot_dfa_by_freq_sets_groups_wide(df, data_points=True):

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("whitegrid")
    sns.set_context("talk")

    # ===============================
    # 🎨 COLORS
    # ===============================
    group_colors = {
        "pink": "#FFC0CB",
        "white": "#D3D3D3",
        "static": "#4F4F4F",
    }

    # ===============================
    # 🏷️ LABELS & ORDER
    # ===============================
    group_label_map = {
        "static": "Non-variable",
        "pink": "Structured",
        "white": "Non-structured",
    }

    groups_order = ["static", "pink", "white"]

    # -------------------------------
    # Reshape wide → long
    # -------------------------------
    value_cols = [c for c in df.columns if c.startswith("DFA_travel_distance")]

    df_long = df.melt(
        id_vars=["ID", "Exact ID"],
        value_vars=value_cols,
        var_name="variable",
        value_name="DFA_travel_distance"
    )

    df_long["Set"] = df_long["variable"].str.extract(r"set_(\d)").astype(int)
    df_long["Freq"] = df_long["variable"].str.extract(r"(\d+)$").astype(int)
    df_long = df_long.rename(columns={"ID": "Group"})

    freqs = sorted(df_long["Freq"].unique())
    groups_order = [g for g in groups_order if g in df_long["Group"].unique()]
    palette = {g: group_colors[g] for g in groups_order}

    # -------------------------------
    # Plotting
    # -------------------------------
    fig, axes = plt.subplots(1, len(freqs), figsize=(18, 6), sharey=True)
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        df_f = df_long[df_long["Freq"] == freq]

        sns.boxplot(
            data=df_f,
            x="Set",
            y="DFA_travel_distance",
            hue="Group",
            hue_order=groups_order,
            palette=palette,
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
                palette=palette,
                dodge=True,
                jitter=0.15,
                alpha=0.6,
                linewidth=0.5,
                ax=ax
            )

        ax.set_title(f"Sampling frequency = {freq} Hz")
        ax.set_xlabel("Set")
        ax.set_ylabel("α exponent")
        ax.legend_.remove()

    # -------------------------------
    # Single legend
    # -------------------------------
    handles, _ = axes[0].get_legend_handles_labels()
    legend_labels = [group_label_map[g] for g in groups_order]

    fig.legend(
        handles[:len(groups_order)],
        legend_labels,
        title="Group",
        loc="lower center",
        ncol=len(groups_order),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.ylim(0.35, 1)
    plt.show()


directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Squat Game\Writting\Publication tryouts\Journal of Sports Sciences\Review 1'
os.chdir(directory)
results = pd.read_excel(r'Results DFA 40 20 10.xlsx')
print(results)

plot_dfa_by_freq_sets_groups_wide(results, data_points=False)
