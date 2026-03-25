import os.path
import json
import datetime
import glob

import numpy as np
import pandas as pd
from datasets import tqdm

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simplify_name(name):
    if name == "baseline":
        return name
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else parts[0]


def plot_performance_metrics(data, show_values=True, gender_specific=False, suffix=""):
    # Preprocess the data
    # if gender_specific:
    #     # add gender specific suffix
    #     data = data.replace(".csv", "_gender_specific.csv")
    df = pd.read_csv(data)
    df["Model"] = df["Embedding"].apply(
        lambda x: (
            "Baseline"
            if x == "baseline"
            else "Combined" if "combined" in x else "Individual"
        )
    )

    # Define metrics and their corresponding column names
    metrics = {"AUC": "Test AUC", "AUPRC": "Test AUPRC", "F1": "Test F1"}

    # Set up the plot style
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#f0f0f0"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1

    # Define a custom color palette (colorblind-friendly)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Function to create plot for a specific gender or overall
    def create_plot(data, gender=None):
        for metric, column in metrics.items():
            fig, ax = plt.subplots(figsize=(24, 12))

            data = data[~data["Target"].isin(["age", "BMI"])]

            # Get unique targets and sort them by prevalence
            targets_prevalence = (
                data.groupby("Target")["Prevalence"].mean().sort_values(ascending=False)
            )
            targets = targets_prevalence.index

            # Increase spacing between target groups
            group_width = 4  # Increased from 3 to 4 for more space
            bar_width = 0.8

            # Plot bars for each target
            for i, target in enumerate(targets):
                target_data = data[data["Target"] == target]

                baseline = target_data[target_data["Embedding"] == "baseline"][
                    column
                ].values[0]
                baseline_name = "baseline"

                combined_data = target_data[target_data["Model"] == "Combined"]
                combined = combined_data[column].max()
                combined_name = simplify_name(
                    combined_data.loc[combined_data[column].idxmax(), "Embedding"]
                )

                individual_data = target_data[target_data["Model"] == "Individual"]
                individual = individual_data[column].max()
                individual_name = simplify_name(
                    individual_data.loc[individual_data[column].idxmax(), "Embedding"]
                )

                # Adjust values for AUC metric
                if metric == "AUC":
                    baseline = max(0, baseline - 0.5)
                    combined = max(0, combined - 0.5)
                    individual = max(0, individual - 0.5)

                x = i * group_width
                ax.bar(
                    x,
                    baseline,
                    color=colors[0],
                    width=bar_width,
                    label="Baseline" if i == 0 else "",
                    alpha=0.8,
                )
                ax.bar(
                    x + 1,
                    combined,
                    color=colors[1],
                    width=bar_width,
                    label="Best Combined" if i == 0 else "",
                    alpha=0.8,
                )
                ax.bar(
                    x + 2,
                    individual,
                    color=colors[2],
                    width=bar_width,
                    label="Best Individual" if i == 0 else "",
                    alpha=0.8,
                )

                # Add labels on top of each bar
                for j, (value, name) in enumerate(
                    zip(
                        [baseline, combined, individual],
                        [baseline_name, combined_name, individual_name],
                    )
                ):
                    if metric == "AUC":
                        label_value = value + 0.5
                    else:
                        label_value = value

                    label = f"{label_value:.2f}" if show_values else name
                    ax.text(
                        x + j,
                        value,
                        label,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )

            # Customize the plot
            ax.set_xlabel("Targets", fontsize=12)
            ax.set_ylabel(f"{metric} Score", fontsize=12)
            title = f"{metric} Scores for Different Models Across Targets {suffix}"
            if gender:
                title += f" ({gender})"
            ax.set_title(title, fontsize=16)

            # Set x-ticks and labels with 45-degree rotation
            ax.set_xticks([i * group_width + 1 for i in range(len(targets))])
            ax.set_xticklabels(
                [
                    f"{target} ({prevalence:.2%})"
                    for target, prevalence in targets_prevalence.items()
                ],
                rotation=45,
                ha="right",
                fontsize=8,
            )

            ax.legend(loc="upper right")

            # Adjust y-axis for AUC plot
            if metric == "AUC":
                ax.set_ylim(0, 0.5)
                yticks = np.arange(0, 0.55, 0.1)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{y + 0.5:.1f}" for y in yticks])

            # Improve overall appearance
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", length=0)

            # Adjust bottom margin to accommodate rotated labels
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Save the figure
            filename = f"{metric}_performance_comparison_seed_{SEED}{suffix}"
            if gender:
                filename += f"_{gender.lower()}"
            plt.savefig(
                os.path.join(FIGS_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight"
            )
            plt.show()
            plt.close()

    if gender_specific and "Gender" in df.columns:
        for gender in df["Gender"].unique():
            create_plot(df[df["Gender"] == gender], gender)
    else:
        create_plot(df)


def plot_performance_metrics_no_combined(
    data, show_values=True, gender_specific=False, suffix="", with_f1_auprc=False
):
    # Preprocess the data
    # if gender_specific:
    #     # add gender specific suffix
    #     data = data.replace(".csv", "_gender_specific.csv")
    df = pd.read_csv(data)
    df["Model"] = df["Embedding"].apply(
        lambda x: "Baseline" if x == "baseline" else "Individual"
    )

    # Define metrics and their corresponding column names
    if with_f1_auprc:
        metrics = {"AUC": "Test AUC", "AUPRC": "Test AUPRC", "F1": "Test F1"}
    else:
        metrics = {"AUC": "Test AUC"}

    # Set up the plot style
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#f0f0f0"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1

    # Define a custom color palette (colorblind-friendly)
    colors = ["#1f77b4", "#ff7f0e"]

    # Function to create plot for a specific gender or overall
    def create_plot(data, gender=None):
        for metric, column in metrics.items():
            fig, ax = plt.subplots(figsize=(24, 12))

            # remove age and BMI rows from the data
            data = data[~data["Target"].isin(["age", "BMI"])]

            # Get unique targets and sort them by prevalence
            targets_prevalence = (
                data.groupby("Target")["Prevalence"].mean().sort_values(ascending=False)
            )
            targets = targets_prevalence.index

            # Increase spacing between target groups
            group_width = 3  # Adjusted for two bars instead of three
            bar_width = 0.8

            # Plot bars for each target
            for i, target in enumerate(targets):
                target_data = data[data["Target"] == target]

                baseline = target_data[target_data["Embedding"] == "baseline"][
                    column
                ].values[0]
                baseline_name = "baseline"

                individual_data = target_data[target_data["Model"] == "Individual"]
                individual = individual_data[column].max()
                individual_name = simplify_name(
                    individual_data.loc[individual_data[column].idxmax(), "Embedding"]
                )

                # Adjust values for AUC metric
                if metric == "AUC":
                    baseline = max(0, baseline - 0.5)
                    individual = max(0, individual - 0.5)

                x = i * group_width
                ax.bar(
                    x,
                    baseline,
                    color=colors[0],
                    width=bar_width,
                    label="Baseline" if i == 0 else "",
                    alpha=0.8,
                )
                ax.bar(
                    x + 1,
                    individual,
                    color=colors[1],
                    width=bar_width,
                    label="Best Individual" if i == 0 else "",
                    alpha=0.8,
                )

                # Add labels on top of each bar
                for j, (value, name) in enumerate(
                    zip(
                        [baseline, individual],
                        [baseline_name, individual_name],
                    )
                ):
                    if metric == "AUC":
                        label_value = value + 0.5
                    else:
                        label_value = value

                    label = f"{label_value:.2f}" if show_values else name
                    ax.text(
                        x + j,
                        value,
                        label,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )

            # Customize the plot
            ax.set_xlabel("Targets", fontsize=12)
            ax.set_ylabel(f"{metric} Score", fontsize=12)
            title = f"{metric} Scores for Different Models Across Targets {suffix}"
            if gender:
                title += f" ({gender})"
            ax.set_title(title, fontsize=16)

            # Set x-ticks and labels with 45-degree rotation
            ax.set_xticks([i * group_width + 0.5 for i in range(len(targets))])
            ax.set_xticklabels(
                [
                    f"{target} ({prevalence:.2%})"
                    for target, prevalence in targets_prevalence.items()
                ],
                rotation=45,
                ha="right",
                fontsize=8,
            )

            ax.legend(loc="upper right")

            # Adjust y-axis for AUC plot
            if metric == "AUC":
                ax.set_ylim(0, 0.5)
                yticks = np.arange(0, 0.55, 0.1)
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{y + 0.5:.1f}" for y in yticks])

            # Improve overall appearance
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", length=0)

            # Adjust bottom margin to accommodate rotated labels
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Save the figure
            filename = f"{metric}_performance_comparison_seed_{SEED}{suffix}"
            if gender:
                filename += f"_{gender.lower()}"
            plt.savefig(
                os.path.join(FIGS_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight"
            )
            plt.show()
            plt.close()

    if gender_specific and "Gender" in df.columns:
        for gender in df["Gender"].unique():
            create_plot(df[df["Gender"] == gender], gender)
    else:
        create_plot(df)


def plot_performance_metrics_w_R2(
    data, show_values=True, gender_specific=False, suffix=""
):
    # Preprocess the data
    df = pd.read_csv(data)
    df["Model"] = df["Embedding"].apply(
        lambda x: (
            "Baseline"
            if x == "baseline"
            else "Combined" if "combined" in x else "Individual"
        )
    )

    # Define metrics and their corresponding column names
    classification_metrics = {"AUC": "Test AUC", "AUPRC": "Test AUPRC", "F1": "Test F1"}
    regression_metrics = {"R2": "Test R2"}

    # Set up the plot style
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#f0f0f0"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "white"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1

    # Define a custom color palette (colorblind-friendly)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    def create_plot(data, gender=None):
        # Get unique targets and sort them by prevalence
        targets_prevalence = (
            data.groupby("Target")["Prevalence"].mean().sort_values(ascending=False)
        )
        targets = targets_prevalence.index

        fig, ax = plt.subplots(figsize=(20, 10))

        bar_width = 0.25
        index = np.arange(len(targets))

        for i, model in enumerate(["Baseline", "Combined", "Individual"]):
            model_data = data[data["Model"] == model]
            values = []

            for target in targets:
                target_data = model_data[model_data["Target"] == target]
                if target in ["age", "BMI"]:
                    metric_value = target_data["Test R2"].values[0]
                else:
                    metric_value = target_data["Test AUC"].values[0]
                    metric_value = max(0, metric_value - 0.5)  # Adjust AUC values
                values.append(metric_value)

            bars = ax.bar(
                index + i * bar_width,
                values,
                bar_width,
                label=model,
                color=colors[i],
                alpha=0.8,
            )

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        # Customize the plot
        ax.set_xlabel("Targets", fontsize=12)
        ax.set_ylabel("Performance Score", fontsize=12)
        title = f"Performance Metrics Across Targets and Models {suffix}"
        if gender:
            title += f" ({gender})"
        ax.set_title(title, fontsize=16)

        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(
            [
                f"{target}\n({prevalence:.2%})"
                for target, prevalence in targets_prevalence.items()
            ],
            rotation=45,
            ha="right",
        )

        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)])

        # Improve overall appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Adjust layout and save figure
        plt.tight_layout()
        filename = f"overall_performance_comparison_seed_{SEED}{suffix}"
        if gender:
            filename += f"_{gender.lower()}"
        plt.savefig(
            os.path.join(FIGS_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight"
        )
        plt.show()
        plt.close()

    if gender_specific and "Gender" in df.columns:
        for gender in df["Gender"].unique():
            create_plot(df[df["Gender"] == gender], gender)
    else:
        create_plot(df)


def plot_target_specific_aucs(data, target_names=None):
    """
    Plot AUC scores for different embeddings, showing male and female comparisons side by side.

    Args:
        data: Path to CSV file or DataFrame
        target_names: List of specific targets to plot (optional)
    """
    # Preprocess the data
    df = pd.read_csv(data) if isinstance(data, str) else data.copy()

    if 'Gender' not in df.columns:
        raise ValueError("Gender column not found in the data")

    # Set up the plot style
    plt.style.use("ggplot")

    # Get unique targets and embeddings
    targets = df["Target"].unique()

    # Filter targets if target_names is provided
    if target_names is not None:
        targets = [t for t in targets if t in target_names]
        if not targets:
            print("None of the specified targets were found in the data.")
            return

    all_embeddings = df["Embedding"].unique()

    # Create a colormap using the new method
    cmap = plt.colormaps["tab20"]

    for target in tqdm(targets):
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (gender, ax) in enumerate([('male', ax1), ('female', ax2)]):
            # Filter data for current gender
            gender_data = df[(df["Target"] == target) & (df["Gender"].str.lower() == gender)]

            if gender_data.empty:
                print(f"No data found for {gender} and target {target}")
                continue

            prevalence = gender_data["Prevalence"].iloc[0]

            # Prepare data for plotting
            embeddings = []
            aucs = []
            for emb in all_embeddings:
                if emb in gender_data["Embedding"].values:
                    embeddings.append(emb)
                    aucs.append(
                        gender_data[gender_data["Embedding"] == emb]["Test AUC"].values[0]
                    )

            # Adjust AUC values
            adjusted_aucs = np.maximum(np.array(aucs) - 0.5, 0)

            # Plot AUC bars with different colors
            bars = ax.bar(
                range(len(embeddings)),
                adjusted_aucs,
                color=[cmap(i) for i in np.linspace(0, 1, len(embeddings))],
            )

            # Customize the plot
            ax.set_xlabel("Embeddings", fontsize=12)
            ax.set_ylabel("AUC Score", fontsize=12)

            ax.set_title(
                f"AUC Scores - {target} ({gender.capitalize()}) [Prevalence: {prevalence:.3f}]",
                fontsize=12,
            )

            ax.set_xticks(range(len(embeddings)))
            ax.set_xticklabels(embeddings, rotation=45, ha="right")

            # Adjust y-axis
            ax.set_ylim(0, 0.5)
            yticks = np.arange(0, 0.55, 0.1)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y + 0.5:.1f}" for y in yticks])

            # Add value labels on top of each bar
            for i, v in enumerate(aucs):
                ax.text(
                    i, adjusted_aucs[i], f"{v:.3f}", ha="center", va="bottom", fontsize=8
                )

        plt.tight_layout()

        # Save the figure with both gender plots
        plt.savefig(
            os.path.join(FIGS_DIR, f"{target}_AUC_comparison_by_gender_{SEED}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


import os
import glob
from scipy import stats
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import multipletests


def get_predictions_data(prediction_results_dir, target_dir, seed):
    """
    Load and combine prediction data from all embedding files using joins to maintain alignment.
    """
    predictions_dfs = {}
    combined_df = None

    # Find all prediction files in this target directory
    pred_files = glob.glob(os.path.join(prediction_results_dir, target_dir, f'combined_predictions_*_{seed}.csv'))

    for file in pred_files:
        # Extract embedding type from filename
        embedding = os.path.basename(file).replace('combined_predictions_', '').replace('.csv', '')
        if 'combined' in embedding:
            embedding = embedding.replace(f'_combined_{seed}', '')
        elif embedding == f'baseline_{seed}':
            embedding = 'baseline'

        # Read predictions file, keeping the index
        df = pd.read_csv(file, index_col=0)

        # Rename columns to include embedding type
        df = df.rename(columns={
            'true_values': f'true_values_{embedding}',
            'predictions': f'predictions_{embedding}',
            'fold': f'fold_{embedding}'
        })

        if combined_df is None:
            combined_df = df
        else:
            # Inner join with existing data
            combined_df = pd.merge(combined_df, df, left_index=True, right_index=True)

    # Split back into the original format
    for col in combined_df.columns:
        if 'true_values_' in col:
            if col == 'true_values_baseline':
                embedding = 'baseline'
            else:
                embedding = col.replace('true_values_embedding_', '')
            predictions_dfs[embedding] = {
                'true_values': combined_df[
                    f'true_values_{"embedding_" if embedding != "baseline" else ""}{embedding}'].values,
                'predictions': combined_df[
                    f'predictions_{"embedding_" if embedding != "baseline" else ""}{embedding}'].values,
                'fold': combined_df[f'fold_{"embedding_" if embedding != "baseline" else ""}{embedding}'].values
            }

    return predictions_dfs


def structural_components(y_true, y_pred):
    """
    Calculate the structural components needed for the DeLong test.
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    X = y_pred[pos_mask]
    Y = y_pred[neg_mask]

    # Calculate structural components
    V10 = np.zeros(len(X))
    for i in range(len(X)):
        V10[i] = np.sum(X[i] > Y) / len(Y)
        V10[i] += np.sum(X[i] == Y) / (2 * len(Y))

    return V10


def delong_test(y_true, y_pred1, y_pred2):
    """
    Implementation of DeLong test for comparing two AUCs.

    References:
    -----------
    DeLong et al. (1988) Comparing the Areas Under Two or More Correlated
    Receiver Operating Characteristic Curves: A Nonparametric Approach
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n1 = np.sum(pos_mask)  # number of positive samples
    n2 = np.sum(neg_mask)  # number of negative samples

    # Calculate structural components for both predictions
    V10_1 = structural_components(y_true, y_pred1)
    V10_2 = structural_components(y_true, y_pred2)

    # Calculate average AUCs
    auc1 = np.mean(V10_1)
    auc2 = np.mean(V10_2)

    # Create combined vector of differences
    D = V10_1 - V10_2

    # Calculate variance of D
    D_mean = np.mean(D)
    S10 = np.var(D, ddof=1) / n1

    # Calculate z-score
    z = D_mean / np.sqrt(S10)

    # Calculate two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    print(f"Debug DeLong test:")
    print(f"AUC1: {auc1:.4f}, AUC2: {auc2:.4f}, Difference: {auc1 - auc2:.4f}")
    print(f"Sample sizes - Positive: {n1}, Negative: {n2}")
    print(f"D mean: {D_mean:.6f}, S10: {S10:.6f}")
    print(f"Z-score: {z:.4f}")
    print(f"P-value: {p_value:.4e}")

    return auc1, auc2, p_value


def compute_fold_based_comparison(data, emb1, emb2):
    """
    Compute statistical comparison between two embeddings considering fold structure.
    Returns the average AUC difference and combined p-value using Fisher's method.
    """
    folds = np.unique(data[emb1]['fold'])
    fold_results = []
    auc_diffs = []
    aucs1 = []
    aucs2 = []

    # Compare within each fold
    for fold in folds:
        # Get fold-specific data
        mask1 = data[emb1]['fold'] == fold
        mask2 = data[emb2]['fold'] == fold
        y1_true = data[emb1]['true_values'][mask1]
        y1_pred = data[emb1]['predictions'][mask1]
        y2_true = data[emb2]['true_values'][mask2]
        y2_pred = data[emb2]['predictions'][mask2]
        # Ensure we're comparing the same samples within fold
        assert np.array_equal(y1_true, y2_true), f"True values don't match in fold {fold}"
        # Compute DeLong test for this fold
        auc1, auc2, p_value = delong_test(y1_true, y1_pred, y2_pred)
        fold_results.append(p_value)
        aucs1.append(auc1)
        aucs2.append(auc2)
        auc_diffs.append(auc2 - auc1)

    # Combine p-values using Fisher's method
    chi_square_stat = -2 * np.sum(np.log(fold_results))
    combined_p_value = 1 - stats.chi2.cdf(chi_square_stat, df=2 * len(folds))
    # Average AUC difference across folds
    mean_auc1 = np.mean(aucs1)
    mean_auc2 = np.mean(aucs2)
    mean_auc_diff = np.mean(auc_diffs)
    std_auc_diff = np.std(auc_diffs)
    print(f"\nOverall comparison {emb1} vs {emb2}:")
    print(f"Mean AUC1: {mean_auc1:.4f}, Mean AUC2: {mean_auc2:.4f}")
    print(f"Mean AUC diff: {mean_auc_diff:.4f} ± {std_auc_diff:.4f}")
    print(f"Per-fold p-values: {', '.join([f'{p:.4e}' for p in fold_results])}")
    print(f"Combined p-value: {combined_p_value:.4e}")
    return mean_auc_diff, std_auc_diff, combined_p_value, auc_diffs


def combine_pvalues_fisher(pvalues):
    """
    Combine p-values using Fisher's method.

    Fisher's method combines p-values from k independent tests by:
    1. Computing -2 * sum(ln(p)) for all p-values
    2. Comparing to chi-square distribution with 2k degrees of freedom

    Args:
        pvalues (list): List of p-values to combine

    Returns:
        float: Combined p-value
    """
    # Handle edge cases
    if not pvalues:
        return 1.0
    if len(pvalues) == 1:
        return pvalues[0]

    # Convert to numpy array and ensure valid p-values
    pvalues = np.array(pvalues)
    pvalues = np.clip(pvalues, 1e-300, 1.0)  # Prevent numerical issues with very small p-values

    # Calculate Fisher's statistic: -2 * sum(ln(p))
    fisher_stat = -2 * np.sum(np.log(pvalues))

    # Get degrees of freedom: 2k where k is number of tests
    df = 2 * len(pvalues)

    # Calculate combined p-value from chi-square distribution
    combined_p = 1 - stats.chi2.cdf(fisher_stat, df)

    return combined_p


def create_single_plot(data, predictions_data, target, metric, output_dir, gender=None, prevalence=None):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.6])

    categories = {
        'Baseline': ['baseline', 'embedding_mfcc'],
        'Speaker Diarization': ['embedding_VoxCeleb', 'embedding_wavLMSD'],
        'Speaker Identification': ['embedding_xVector', 'embedding_VoxCelebFintuned', 'embedding_EffNet'],
        'Pretrained': ['embedding_wav2vec2Base', 'embedding_wav2vec2Large', 'embedding_wavLMBase',
                       'embedding_wavLMLarge', 'embedding_wav2vec2XLSRVanilla'],
        'Hebrew': ['embedding_wav2vec2XLSRVFineTuned', 'embedding_wav2vec2HebrewPretrained'],
        'Emotion Recognition': ['embedding_wavlmEM', 'embedding_wav2vecEM'],
        'Ensemble': [config for config in data['embedding'].unique() 
                    if 'top' in str(config) and ('mean' in str(config) or 'median' in str(config))]
    }

    category_colors = {
        'Baseline': '#1f77b4',
        'Speaker Diarization': '#2ca02c',
        'Speaker Identification': '#ff7f0e',
        'Pretrained': '#9467bd',
        'Hebrew': '#d62728',
        'Emotion Recognition': '#c70eb4',
        'Ensemble': '#8c564b'  
    }

    embedding_order = []
    for category in categories.values():
        embedding_order.extend([emb for emb in category if emb in data['embedding'].unique()])

    positions = []
    current_pos = 0
    category_positions = {}
    category_widths = {}

    for category_name, category_embs in categories.items():
        category_embs = [emb for emb in category_embs if emb in data['embedding'].unique()]
        if category_embs:
            category_positions[category_name] = current_pos
            for emb in category_embs:
                positions.append(current_pos)
                current_pos += 1
            category_widths[category_name] = len(category_embs)
            current_pos += 0.8

    performance_data = []
    colors = []
    max_values = []
    for emb in embedding_order:
        emb_data = data[data['embedding'] == emb][metric].values
        performance_data.append(emb_data)
        max_values.append(np.max(emb_data))
        for cat_name, cat_embs in categories.items():
            if emb in cat_embs:
                colors.append(category_colors[cat_name])
                break

    box_plot = ax.boxplot(performance_data, positions=positions[:len(performance_data)],
                          patch_artist=True, showfliers=False)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (pos, perf_data) in enumerate(zip(positions[:len(performance_data)], performance_data)):
        ax.scatter([pos] * len(perf_data), perf_data, color='navy', alpha=0.4, s=20)

    baseline_data = data[data['embedding'] == 'baseline'][metric].values
    baseline_median = np.median(baseline_data)
    ax.axhline(y=baseline_median, color='red', linestyle='--', label='Baseline Median')

    y_min, y_max = ax.get_ylim()
    data_range = y_max - y_min

    box_positions = dict(zip(embedding_order, positions[:len(embedding_order)]))
    annotation_base = np.max(max_values) + data_range * 0.02
    add_significance_annotations(ax, predictions_data, embedding_order, annotation_base, box_positions)

    plt.xticks(positions[:len(embedding_order)], embedding_order, rotation=45, ha='right')

    for category_name, start_pos in category_positions.items():
        width = (category_widths[category_name] - 1) * 0.3  # Adjust for new spacing
        center_pos = start_pos + width / 2
        ax.text(center_pos, annotation_base + data_range * 0.15, category_name,
                horizontalalignment='center',
                fontsize=10, fontweight='bold',
                color=category_colors[category_name])

    full_target_name = get_full_target_name(target)
    title = f'{metric.upper()} for {full_target_name}'
    if gender:
        title += f' ({gender.upper()})'
    if prevalence is not None:
        title += f'\nPrevalence: {prevalence:.1f}%'

    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Embeddings")
    ax.legend()
    ax.set_ylim(y_min, annotation_base + data_range * 0.25)

    filename = f"{target}_{metric}"
    if gender:
        filename += f"_{gender}"
    filename += "_comparison.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def benjamini_hochberg_correction(p_values):
    """
    Apply Benjamini-Hochberg correction to p-values.
    Returns corrected p-values.
    """
    _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    return p_corrected


def compute_baseline_comparisons(data):
    """
    Compute statistical comparisons between baseline and all other embeddings using fold-based comparison.
    Returns dictionary with results and corrected p-values.
    """
    embeddings = sorted([emb for emb in data.keys() if emb != 'baseline'])
    comparison_results = []
    p_values = []

    # Compare each embedding with baseline using fold-based comparison
    for emb in embeddings:
        # Compute fold-based comparison
        mean_auc_diff, std_auc_diff, p_value, auc_diffs = compute_fold_based_comparison(data, 'baseline', emb)

        comparison_results.append({
            'embedding': emb,
            'auc_diff': mean_auc_diff,
            'auc_diff_std': std_auc_diff,
            'p_value': p_value,
            'per_fold_diffs': auc_diffs,
            'baseline_auc': np.mean([roc_auc_score(data['baseline']['true_values'][data['baseline']['fold'] == fold],
                                                 data['baseline']['predictions'][data['baseline']['fold'] == fold])
                                   for fold in np.unique(data['baseline']['fold'])]),
            'embedding_auc': np.mean([roc_auc_score(data[emb]['true_values'][data[emb]['fold'] == fold],
                                                  data[emb]['predictions'][data[emb]['fold'] == fold])
                                    for fold in np.unique(data[emb]['fold'])])
        })
        p_values.append(p_value)

    # Apply Benjamini-Hochberg correction
    corrected_p_values = benjamini_hochberg_correction(p_values)

    # Add corrected p-values to results
    for result, corrected_p in zip(comparison_results, corrected_p_values):
        result['corrected_p_value'] = corrected_p

    return comparison_results


def generate_statistical_report(data, embeddings, target, gender=None):
    """Generate a detailed statistical report comparing each embedding to baseline using fold-based comparison.

    Args:
        data: Dictionary containing prediction data
        embeddings: List of embedding names
        target: Target variable name
        gender: Optional gender specification
    """
    report = []
    report.append(f"\nStatistical Report for {target}" + (f" ({gender})" if gender else ""))
    report.append("=" * 80)

    # Compute comparisons with baseline using fold-based method
    comparison_results = compute_baseline_comparisons(data)

    # Format results
    report.append("\nComparisons with Baseline (Fold-based analysis):")
    report.append("-" * 50)

    for result in comparison_results:
        emb = result['embedding']
        significance = ""
        if result['corrected_p_value'] < 0.001:
            significance = "***"
        elif result['corrected_p_value'] < 0.01:
            significance = "**"
        elif result['corrected_p_value'] < 0.05:
            significance = "*"

        report.append(f"Baseline vs {emb}:")
        report.append(f"{'Baseline AUC':30s}: {result['baseline_auc']:.4f}")
        report.append(f"{f'{emb} AUC':30s}: {result['embedding_auc']:.4f}")
        report.append(f"{'Mean AUC difference':30s}: {result['auc_diff']:+.4f} ± {result['auc_diff_std']:.4f}")
        report.append(f"{'Raw p-value':30s}: {result['p_value']:.4e}")
        report.append(f"{'Corrected p-value':30s}: {result['corrected_p_value']:.4e} {significance}")
        report.append(f"{'Per-fold AUC differences':30s}: {', '.join([f'{d:+.4f}' for d in result['per_fold_diffs']])}")
        report.append("-" * 50)

    report.append("\nSignificance levels after Benjamini-Hochberg correction:")
    report.append("* p < 0.05, ** p < 0.01, *** p < 0.001")
    report.append("\nNote: Statistics are computed using fold-based comparisons")
    return "\n".join(report)


def add_significance_annotations(ax, data, embedding_order, y_max, box_positions):
    """Add significance annotations for comparisons with baseline."""
    comparison_results = compute_baseline_comparisons(data)
    y_offset = y_max * 0.015

    ordered_results = []
    for emb in embedding_order:
        for result in comparison_results:
            if result['embedding'] == emb:
                ordered_results.append(result)
                break

    for result in ordered_results:
        if (result['corrected_p_value'] < 0.05 and
                result['embedding'] in box_positions and
                result['auc_diff'] > 0):  # Only show for positive differences

            emb = result['embedding']
            emb_pos = box_positions[emb]

            # Determine significance level
            sig_symbol = ('***' if result['corrected_p_value'] < 0.001 else
                          '**' if result['corrected_p_value'] < 0.01 else '*')

            # Add larger, bold asterisk
            ax.text(emb_pos, y_max + y_offset, sig_symbol,
                    ha='center', va='bottom', fontsize=14,
                    weight='bold')

            # # Add difference value below asterisk
            # diff_text = f"+{result['auc_diff']:.3f}"  # Force + sign
            # ax.text(emb_pos, y_max, diff_text,
            #         ha='center', va='bottom', fontsize=10)

# def create_performance_plots_from_files(prediction_results_dir, targets=None, output_dir=None, generate_report=False,
#                                         gender_specific=False, seed=42):
#     """
#     Create performance plots using fold-level results from target-specific directories.
#     Includes statistical significance testing between embedding performances.
#
#     Parameters:
#         prediction_results_dir: str
#             Directory containing the prediction results
#         targets: list, optional
#             List of specific targets to process. If None, process all targets
#         output_dir: str, optional
#             Directory to save the output plots. If None, creates a 'plots' subdirectory
#         generate_report: bool, default=False
#             Whether to generate statistical reports
#         gender_specific: bool, default=False
#             If True, process gender-specific directories (male/female)
#             If False, process only the '_all' directories
#     """
#     try:
#         # Set up output directory
#         if output_dir is None:
#             output_dir = os.path.join(prediction_results_dir, 'plots')
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Get all target directories
#         target_dirs = [d for d in os.listdir(prediction_results_dir)
#                       if os.path.isdir(os.path.join(prediction_results_dir, d))]
#
#         # Filter directories based on gender_specific flag
#         if gender_specific:
#             # Include only male and female directories
#             target_dirs = [d for d in target_dirs if d.endswith('_male') or d.endswith('_female')]
#         else:
#             # Include only '_all' directories
#             target_dirs = [d for d in target_dirs if d.endswith('_all')]
#
#         # Filter targets if specified
#         if targets is not None:
#             filtered_dirs = []
#             for t in targets:
#                 filtered_dirs.extend([d for d in target_dirs if t in d])
#             target_dirs = filtered_dirs
#
#             if not target_dirs:
#                 print(f"No directories found for specified targets: {targets}")
#                 return
#
#         # Sort directories for consistent processing
#         target_dirs.sort()
#
#         # Process each target directory
#         for target_dir in tqdm(target_dirs):
#             try:
#                 # Extract target and gender from directory name
#                 parts = target_dir.rsplit('_', 1)
#                 target = parts[0]
#                 gender = parts[1]  # Will be 'all', 'male', or 'female'
#
#                 target_path = os.path.join(prediction_results_dir, target_dir)
#
#                 # Find all fold results files in this target directory
#                 fold_files = glob.glob(os.path.join(target_path, 'fold_level_results_*.csv'))
#
#                 if not fold_files:
#                     print(f"No fold results found in {target_dir}")
#                     continue
#
#                 # Read and combine all fold results for this target
#                 all_results = []
#                 for file in fold_files:
#                     try:
#                         # Extract embedding name from filename
#                         embedding = os.path.basename(file).replace('fold_level_results_', '').replace('.csv', '')
#                         if 'combined' in embedding:
#                             embedding = embedding.replace(f'_combined_{seed}', '')
#                         elif embedding == f'baseline_{seed}':
#                             embedding = 'baseline'
#
#                         df = pd.read_csv(file)
#                         df['embedding'] = embedding
#                         all_results.append(df)
#                     except Exception as e:
#                         print(f"Error processing file {file}: {str(e)}")
#                         continue
#
#                 if not all_results:
#                     print(f"No valid results found in {target_dir}")
#                     continue
#
#                 combined_results = pd.concat(all_results, ignore_index=True)
#
#                 # Determine if regression or classification
#                 is_regression = combined_results['Is Regression'].iloc[0]
#                 metric = 'r2' if is_regression else 'auc'
#
#                 try:
#                     # Get predictions data for statistical testing
#                     predictions_data = get_predictions_data(prediction_results_dir, target_dir, seed=seed)
#
#                     # Calculate mean prevalence for classification tasks
#                     mean_prevalence = combined_results['prevalence'].mean() * 100 if not is_regression else None
#
#                     # Create plot with significance annotations
#                     create_single_plot(
#                         combined_results,
#                         predictions_data,
#                         target,
#                         metric,
#                         output_dir,
#                         gender=None if gender == 'all' else gender,
#                         prevalence=mean_prevalence
#                     )
#
#                     # Generate statistical report if requested
#                     if generate_report and not is_regression:
#                         try:
#                             report = generate_statistical_report(predictions_data, sorted(predictions_data.keys()), target, gender)
#
#                             # Save report to file
#                             report_filename = f"{target}_statistical_report"
#                             if gender != 'all':
#                                 report_filename += f"_{gender}"
#                             report_filename += ".txt"
#
#                             report_path = os.path.join(output_dir, report_filename)
#                             with open(report_path, 'w') as f:
#                                 f.write(report)
#                             print(report)  # Also print to console
#                         except ValueError as ve:
#                             print(f"Error generating statistical report for {target_dir}: {str(ve)}")
#                         except Exception as e:
#                             print(f"Unexpected error generating statistical report for {target_dir}: {str(e)}")
#
#                 except ValueError as ve:
#                     print(f"Error processing predictions for {target_dir}: {str(ve)}")
#                     print("Skipping this target directory...")
#                     continue
#                 except Exception as e:
#                     print(f"Unexpected error processing {target_dir}: {str(e)}")
#                     continue
#
#             except Exception as e:
#                 print(f"Error processing directory {target_dir}: {str(e)}")
#                 continue
#
#     except Exception as e:
#         print(f"Fatal error in create_performance_plots_from_files: {str(e)}")
#         raise

def create_main_radar_plot(data, model_categories, condition_categories, output_dir, metric='auc'):
    """Create main radar plot showing best performance per model category for all medical conditions.
    For AUC metric, values below 0.5 are clipped to 0.5."""
    for gender in ['male', 'female']:
        gender_data = data[data['Gender'] == gender]
        if gender_data.empty:
            print(f'Skipped {gender} plot due to no gender data')
            continue

        best_performances = {}
        best_models = {}
        conditions_order = []
        category_colors = plt.cm.tab20(np.linspace(0, 1, len(condition_categories)))

        # Process conditions and find best performances
        for med_category, conditions in condition_categories.items():
            present_conditions = [c for c in conditions if c in gender_data['target'].unique()]
            if present_conditions:
                conditions_order.extend(present_conditions)

                for condition in present_conditions:
                    condition_data = gender_data[gender_data['target'] == condition]
                    if not condition_data.empty:
                        best_performances[condition] = {}
                        best_models[condition] = {}

                        for model_cat, models in model_categories.items():
                            cat_performances = []
                            for model in models:
                                if model in condition_data['embedding'].unique():
                                    perf = condition_data[condition_data['embedding'] == model][metric].mean()
                                    # Clip AUC values below 0.5 to 0.5
                                    if metric == 'auc':
                                        perf = max(0.5, perf)
                                    cat_performances.append((perf, model))

                            if cat_performances:
                                best_perf, best_model = max(cat_performances, key=lambda x: x[0])
                                best_performances[condition][model_cat] = best_perf
                                best_models[condition][model_cat] = best_model

        if not best_performances:
            continue

        # Increase figure size to accommodate labels
        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(111, projection='polar')

        n_conditions = len(conditions_order)
        angles = [n / float(n_conditions) * 2 * np.pi for n in range(n_conditions)]
        angles += angles[:1]

        # Plot each model category's performance
        model_colors = plt.cm.Set2(np.linspace(0, 1, len(model_categories)))

        for model_cat, color in zip(model_categories.keys(), model_colors):
            values = [best_performances[cond].get(model_cat, 0.5 if metric == 'auc' else 0) for cond in
                      conditions_order]
            values += values[:1]
            ax.plot(angles, values, '-', linewidth=2, label=model_cat, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        condition_colors = []

        # Color code conditions by their category
        prev_angle = 0
        for idx, (med_category, conditions) in enumerate(condition_categories.items()):
            cat_conditions = [c for c in conditions if c in conditions_order]
            if cat_conditions:
                angle = (len(cat_conditions) / n_conditions) * 2 * np.pi
                mid_angle = prev_angle + angle / 2

                # Add separator
                ax.plot([prev_angle, prev_angle], [0.5 if metric == 'auc' else 0.4, 0.9], '--', color='gray', alpha=0.3)

                # Add category label further out
                label_radius = 1.3  # Increased from 1.1
                rotation = 0  # Horizontal text
                ax.text(mid_angle, label_radius, med_category,
                        ha='center', va='center',
                        rotation=rotation,
                        fontsize=10, fontweight='bold',
                        color=category_colors[idx])

                # Store condition colors
                condition_colors.extend([category_colors[idx]] * len(cat_conditions))
                prev_angle += angle

        # Set condition labels with matching colors and position them further out
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Remove default labels

        # Add custom positioned labels
        for idx, (angle, condition) in enumerate(zip(angles[:-1], conditions_order)):
            # Position labels further from the plot
            label_radius = 1.05  # Increased from default
            x = label_radius * np.cos(angle)
            y = label_radius * np.sin(angle)

            # Adjust text alignment based on angle
            rotation = np.degrees(angle)
            if angle >= np.pi / 2 and angle <= 3 * np.pi / 2:
                rotation += 180

            ax.text(angle, label_radius, condition,
                    rotation=rotation,
                    ha='center', va='center',
                    fontsize=8,
                    color=condition_colors[idx])

        # Set y-axis limits based on metric
        if metric == 'auc':
            ax.set_ylim(0.5, 0.9)
            ax.set_rgrids([0.5, 0.6, 0.7, 0.8], angle=0)
        else:
            ax.set_ylim(0.4, 0.9)
            ax.set_rgrids([0.5, 0.6, 0.7, 0.8], angle=0)

        plt.legend(loc='center left', bbox_to_anchor=(1.2, 1.0))
        # plt.title(
        #     f"Best Model Category Performance Across Medical Conditions - {gender.upper()}\nMetric: {metric.upper()}")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"radar_plot_best_performance_{gender}.pdf")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='pdf')
        plt.show()
        plt.close()

    return best_model

def create_category_specific_plots(data, model_categories, condition_categories, output_dir, metric='auc'):
    """Create individual radar plots for each medical category."""
    for gender in ['male', 'female']:
        gender_data = data[data['Gender'] == gender]
        if gender_data.empty:
            continue

        for med_category, conditions in condition_categories.items():
            present_conditions = [c for c in conditions if c in gender_data['target'].unique()]
            if not present_conditions:
                continue

            best_performances = {}
            for condition in present_conditions:
                condition_data = gender_data[gender_data['target'] == condition]
                if not condition_data.empty:
                    best_performances[condition] = {}
                    for model_cat, models in model_categories.items():
                        cat_performances = []
                        for model in models:
                            if model in condition_data['embedding'].unique():
                                perf = condition_data[condition_data['embedding'] == model][metric].mean()
                                cat_performances.append(perf)
                        if cat_performances:
                            best_performances[condition][model_cat] = max(cat_performances)

            if not best_performances:
                continue

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')

            n_conditions = len(present_conditions)
            angles = [n / float(n_conditions) * 2 * np.pi for n in range(n_conditions)]
            angles += angles[:1]

            model_colors = plt.cm.Set2(np.linspace(0, 1, len(model_categories)))
            for model_cat, color in zip(model_categories.keys(), model_colors):
                values = [best_performances[cond].get(model_cat, 0) for cond in present_conditions]
                values += values[:1]
                ax.plot(angles, values, '-', linewidth=2, label=model_cat, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)

            ax.set_xticks(angles[:-1])
            plt.xticks(angles[:-1], present_conditions, fontsize=8)
            ax.set_ylim(0.4, 0.9)
            ax.set_rgrids([0.5, 0.6, 0.7, 0.8], angle=0)

            plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
            plt.title(f"{med_category} - {gender.upper()}\nMetric: {metric.upper()}")

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"radar_plot_{med_category.lower().replace(' ', '_')}_{gender}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()


def create_category_mean_plot(data, model_categories, condition_categories, output_dir, metric='auc'):
    """Create radar plot showing mean performance per model category for each medical category."""
    for gender in ['male', 'female']:
        gender_data = data[data['Gender'] == gender]
        if gender_data.empty:
            continue

        category_means = {}
        for med_category, conditions in condition_categories.items():
            present_conditions = [c for c in conditions if c in gender_data['target'].unique()]
            if not present_conditions:
                continue

            category_means[med_category] = {}
            for model_cat, models in model_categories.items():
                cat_best_perfs = []
                for condition in present_conditions:
                    condition_data = gender_data[gender_data['target'] == condition]
                    if not condition_data.empty:
                        model_perfs = []
                        for model in models:
                            if model in condition_data['embedding'].unique():
                                perf = condition_data[condition_data['embedding'] == model][metric].mean()
                                model_perfs.append(perf)
                        if model_perfs:
                            cat_best_perfs.append(max(model_perfs))
                if cat_best_perfs:
                    category_means[med_category][model_cat] = np.mean(cat_best_perfs)

        if not category_means:
            continue

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')

        categories = list(category_means.keys())
        n_categories = len(categories)
        angles = [n / float(n_categories) * 2 * np.pi for n in range(n_categories)]
        angles += angles[:1]

        model_colors = plt.cm.Set2(np.linspace(0, 1, len(model_categories)))
        for model_cat, color in zip(model_categories.keys(), model_colors):
            values = [category_means[cat].get(model_cat, 0) for cat in categories]
            values += values[:1]
            ax.plot(angles, values, '-', linewidth=2, label=model_cat, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        plt.xticks(angles[:-1], categories, fontsize=8)
        ax.set_ylim(0.4, 0.9)
        ax.set_rgrids([0.5, 0.6, 0.7, 0.8], angle=0)

        plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
        plt.title(f"Mean Category Performance - {gender.upper()}\nMetric: {metric.upper()}")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"radar_plot_category_means_{gender}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()


def create_performance_plots_from_files(prediction_results_dir, targets=None, output_dir=None, generate_report=False,
                                        gender_specific=False, create_radar=False, seed=42):
    if output_dir is None:
        output_dir = os.path.join(prediction_results_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    model_categories = {
        'Baseline': ['baseline'],
        'MFCC': ['embedding_mfcc'],
        'Speaker Diarization': ['embedding_VoxCeleb', 'embedding_wavLMSD'],
        'Speaker Identification': ['embedding_xVector', 'embedding_VoxCelebFintuned', 'embedding_EffNet'],
        'SOTA Speech Foundational Models': ['embedding_wav2vec2Base', 'embedding_wav2vec2Large', 'embedding_wavLMBase',
                       'embedding_wavLMLarge', 'embedding_wav2vec2XLSRVanilla'],
        'Hebrew': ['embedding_wav2vec2XLSRVFineTuned', 'embedding_wav2vec2HebrewPretrained'],
        'Emotion Recognition': ['embedding_wavlmEM', 'embedding_wav2vecEM'],
        'Ensemble': [config for config in data['embedding'].unique() 
                    if 'top' in str(config) and ('mean' in str(config) or 'median' in str(config))]
    }

    # Define medical condition categories
    condition_categories = {
        'Mental Health': [
            'Anxiety or fear-related disorders',
            'Depressive disorders',
            'Mood and anxiety disorders'
        ],
        'Sleep': [
            'Sleep disorders',
            'Obstructive sleep apnoea',
            'has_SA'
        ],
        'Respiratory': [
            'Respiratory disorders',
            'Asthma',
            'current_smoker',
            'past_smoker'
        ],
        'ENT': [
            'Rhinitis and sinusitis',
            'Chronic rhinosinusitis'
        ],
        'Neurological': [
            'Headache and migraine disorders',
            'Migraine'
        ],
        'Endocrine & Other': [
            'Thyroid disorders',
            'hypothyroidism',
            'COVID-19'
        ]
    }

    target_dirs = [d for d in os.listdir(prediction_results_dir)
                   if os.path.isdir(os.path.join(prediction_results_dir, d))]

    if gender_specific:
        target_dirs = [d for d in target_dirs if d.endswith('_male') or d.endswith('_female')]
    else:
        target_dirs = [d for d in target_dirs if d.endswith('_all')]

    if targets is not None:
        filtered_dirs = []
        for t in targets:
            filtered_dirs.extend([d for d in target_dirs if t in d])
        target_dirs = filtered_dirs

    target_dirs.sort()

    # If create_radar is True, only create the radar plot
    if create_radar:
        all_results = []
        for target_dir in tqdm(target_dirs):
            try:
                parts = target_dir.rsplit('_', 1)
                target = parts[0]
                target_path = os.path.join(prediction_results_dir, target_dir)
                fold_files = glob.glob(os.path.join(target_path, 'fold_level_results_*.csv'))

                if not fold_files:
                    continue

                dir_results = []
                for file in fold_files:
                    try:
                        embedding = os.path.basename(file).replace('fold_level_results_', '').replace('.csv', '')
                        if 'combined' in embedding:
                            embedding = embedding.replace(f'_combined_{seed}', '')
                        elif embedding == f'baseline_{seed}':
                            embedding = 'baseline'

                        df = pd.read_csv(file)
                        df['embedding'] = embedding
                        df['target'] = target
                        dir_results.append(df)
                    except Exception as e:
                        print(f"Error processing file {file}: {str(e)}")
                        raise

                if dir_results:
                    combined_results = pd.concat(dir_results, ignore_index=True)
                    all_results.append(combined_results)

            except Exception as e:
                print(f"Error processing directory {target_dir}: {str(e)}")
                raise

        if all_results:
            try:
                combined_data = pd.concat(all_results, ignore_index=True)
                is_regression = combined_data['Is Regression'].iloc[0]
                metric = 'r2' if is_regression else 'auc'
                # create_category_radar_plots(combined_data, model_categories, output_dir, metric=metric)

                # Create all plots
                best_models = create_main_radar_plot(combined_data, model_categories, condition_categories, output_dir)
                create_category_specific_plots(combined_data, model_categories, condition_categories, output_dir)
                create_category_mean_plot(combined_data, model_categories, condition_categories, output_dir)
            except Exception as e:
                print(f"Error creating radar plot: {str(e)}")
                raise
        return

    # Original plot creation code for individual plots when create_radar is False
    for target_dir in tqdm(target_dirs):
        try:
            parts = target_dir.rsplit('_', 1)
            target = parts[0]
            gender = parts[1]

            target_path = os.path.join(prediction_results_dir, target_dir)
            fold_files = glob.glob(os.path.join(target_path, 'fold_level_results_*.csv'))

            if not fold_files:
                continue

            dir_results = []
            for file in fold_files:
                try:
                    embedding = os.path.basename(file).replace('fold_level_results_', '').replace('.csv', '')
                    if 'combined' in embedding:
                        embedding = embedding.replace(f'_combined_{seed}', '')
                    elif embedding == f'baseline_{seed}':
                        embedding = 'baseline'

                    df = pd.read_csv(file)
                    df['embedding'] = embedding
                    dir_results.append(df)
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue

            if not dir_results:
                continue

            combined_results = pd.concat(dir_results, ignore_index=True)
            is_regression = combined_results['Is Regression'].iloc[0]
            metric = 'r2' if is_regression else 'auc'

            try:
                predictions_data = get_predictions_data(prediction_results_dir, target_dir, seed=seed)
                mean_prevalence = combined_results['prevalence'].mean() * 100 if not is_regression else None

                create_single_plot(
                    combined_results,
                    predictions_data,
                    target,
                    metric,
                    output_dir,
                    gender=None if gender == 'all' else gender,
                    prevalence=mean_prevalence
                )

                if generate_report and not is_regression:
                    try:
                        report = generate_statistical_report(predictions_data, sorted(predictions_data.keys()), target,
                                                             gender)
                        report_filename = f"{target}_statistical_report"
                        if gender != 'all':
                            report_filename += f"_{gender}"
                        report_filename += ".txt"
                        report_path = os.path.join(output_dir, report_filename)
                        with open(report_path, 'w') as f:
                            f.write(report)
                    except Exception as e:
                        print(f"Error generating statistical report for {target_dir}: {str(e)}")

            except Exception as e:
                print(f"Error processing {target_dir}: {str(e)}")
                raise

        except Exception as e:
            print(f"Error processing directory {target_dir}: {str(e)}")
            raise



# =============================================================================
# Analysis and Plotting Function
# =============================================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

def create_single_plot_new(data, wilcoxon_results, target, metric, output_dir,
                           gender=None, prevalence=None):
    """Create a boxplot for a single target."""
    # Model categories and colors.
    categories = {
        'Baseline': ['baseline', 'embedding_mfcc'],
        'Speaker Diarization': ['embedding_VoxCeleb', 'embedding_wavLMSD'],
        'Speaker Identification': ['embedding_xVector', 'embedding_EffNet', 'embedding_VoxCelebFintuned'],
        'Pretrained': ['embedding_wav2vec2Base', 'embedding_wav2vec2Large', 'embedding_wavLMBase',
                       'embedding_wavLMLarge', 'embedding_wav2vec2XLSRVanilla'],
        'Hebrew': ['embedding_wav2vec2XLSRVFineTuned', 'embedding_wav2vec2HebrewPretrained'],
        'Emotion Recognition': ['embedding_wavlmEM', 'embedding_wav2vecEM'],
        'Ensemble': [emb for emb in data['embedding'].unique() if 'top' in str(emb)]
    }
    category_colors = {
        'Baseline': '#1f77b4',
        'Speaker Diarization': '#2ca02c',
        'Speaker Identification': '#ff7f0e',
        'Pretrained': '#9467bd',
        'Hebrew': '#d62728',
        'Emotion Recognition': '#c70eb4',
        'Ensemble': '#8c564b'  # Brown color for ensemble models
    }
    
    # Determine the order of embeddings for plotting based on categories.
    embedding_order = []
    for cat_name, emb_list in categories.items():
        for emb in emb_list:
            if emb in data['embedding'].unique():
                embedding_order.append(emb)

    # Compute x-axis positions for each embedding (with extra spacing between categories).
    positions = []
    current_pos = 0
    category_positions = {}   # Starting position for each category group.
    category_widths = {}      # Number of embeddings in each category.
    box_positions = {}        # Mapping from embedding to its x position.

    for cat_name, emb_list in categories.items():
        valid_embs = [emb for emb in emb_list if emb in data['embedding'].unique()]
        if valid_embs:
            category_positions[cat_name] = current_pos
            for emb in valid_embs:
                positions.append(current_pos)
                box_positions[emb] = current_pos
                current_pos += 1
            category_widths[cat_name] = len(valid_embs)
            current_pos += 0.8  # Extra spacing between categories.

    # Prepare performance data and colors.
    performance_data = []
    colors = []
    for emb in embedding_order:
        auc_values = data.loc[data['embedding'] == emb, 'auc'].values
        performance_data.append(auc_values)
        # Get the color based on the embedding's category.
        emb_color = None
        for cat_name, emb_list in categories.items():
            if emb in emb_list:
                emb_color = category_colors[cat_name]
                break
        colors.append(emb_color)

    # Create the figure.
    fig, ax = plt.subplots(figsize=(20, 12))

    # Draw the boxplots.
    bp = ax.boxplot(performance_data, positions=positions[:len(performance_data)],
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual seed data points.
    for pos, auc_vals in zip(positions[:len(performance_data)], performance_data):
        ax.scatter([pos] * len(auc_vals), auc_vals, color='navy', alpha=0.6, s=30)

    # Draw a horizontal line at the baseline median (if baseline exists).
    baseline_median = None
    if 'baseline' in data['embedding'].unique():
        baseline_vals = data.loc[data['embedding'] == 'baseline', 'auc'].values
        baseline_median = np.median(baseline_vals)
        ax.axhline(y=baseline_median, color='red', linestyle='--', label='Baseline Median')

    # Get the y-axis limits and range.
    y_min, y_max = ax.get_ylim()
    data_range = y_max - y_min

    # Annotate significant differences with an asterisk.
    # Only annotate if the Wilcoxon test is significant (p < 0.05) and the embedding's median is higher than baseline.
    for emb in embedding_order:
        if emb in wilcoxon_results:
            p_value = wilcoxon_results[emb].get('p_value', None)
            if p_value is not None and p_value < 0.05:
                emb_values = data.loc[data['embedding'] == emb, 'auc'].values
                median_emb = np.median(emb_values) if len(emb_values) > 0 else None
                if baseline_median is None or (median_emb is not None and median_emb > baseline_median):
                    max_val = np.max(emb_values) if len(emb_values) > 0 else y_max
                    offset = data_range * 0.05
                    x_pos = box_positions[emb]
                    ax.text(x_pos, max_val + offset, '*', ha='center', va='bottom',
                            color='black', fontsize=20, fontweight='bold')

    # Set x-tick labels.
    ax.set_xticks(positions[:len(embedding_order)])
    ax.set_xticklabels(embedding_order, rotation=45, ha='right', fontsize=10)

    # Add category labels above the groups.
    for cat_name, start_pos in category_positions.items():
        width = category_widths[cat_name]
        center = start_pos + (width - 1) / 2.0
        ax.text(center, y_max + data_range * 0.05, cat_name,
                horizontalalignment='center', fontsize=12, fontweight='bold',
                color=category_colors[cat_name])

    # Compose the title.
    full_target_name = target  # Replace with a helper if needed.
    title = f'{metric.upper()} for {full_target_name}'
    if gender:
        title += f' ({gender.upper()})'
    if prevalence is not None:
        title += f'\nPrevalence: {prevalence:.1f}%'
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(metric.upper(), fontsize=14)
    ax.set_xlabel("Embeddings", fontsize=14)
    ax.legend()

    # Adjust the y-limit to leave room for annotations.
    ax.set_ylim(y_min, y_max + data_range * 0.15)

    # Save the figure with explicit background color
    fig.patch.set_facecolor('white')  # Set figure background to white
    ax.set_facecolor('white')  # Set axis background to white
    
    # Add timestamp to filename for uniqueness
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{target}_{metric}"
    if gender:
        filename += f"_{gender}"
    filename += f"_comparison_{timestamp}.png"
    
    plot_path = os.path.join(output_dir, filename)
    print(f"Saving plot to: {plot_path}")
    
    try:
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Successfully saved plot to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot to {plot_path}: {e}")
        
        # Try saving to home directory as fallback
        try:
            home_dir = os.path.expanduser("~")
            fallback_path = os.path.join(home_dir, filename)
            plt.savefig(fallback_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved plot to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"Error saving to fallback location: {e2}")
    
    # Remove plt.show() as it doesn't work in non-interactive environments
    plt.close(fig)  # Explicitly close the figure


def create_radar_plots_from_combined_results(data, model_categories, condition_categories, output_dir, metric='auc'):
    """
    Create radar plots showing best performance per model category for all medical conditions.
    For the AUC metric, values below 0.5 are clipped to 0.5.

    This function expects `data` to be a DataFrame containing at least:
      - 'Gender': indicating the gender group (e.g., 'male' or 'female')
      - 'target': the medical condition
      - 'embedding': the model/embedding name
      - a column for the performance metric (e.g. 'auc')

    Parameters:
      data (DataFrame): Combined results data.
      model_categories (dict): Dictionary mapping model category names to lists of embedding names.
      condition_categories (dict): Dictionary mapping condition category names to lists of condition names.
      output_dir (str): Directory where the radar plot PDFs will be saved.
      metric (str): Performance metric to plot (default is 'auc').

    Returns:
      best_models (dict): A dictionary of the best model per condition (per gender).
    """
    # Loop over genders.
    for gender in ['male', 'female']:
        gender_data = data[data['Gender'] == gender]
        if gender_data.empty:
            print(f"Skipped {gender} plot due to no data for this gender.")
            continue

        best_performances = {}
        best_models = {}
        conditions_order = []
        # Use a colormap to assign a color to each condition category.
        category_colors = plt.cm.tab20(np.linspace(0, 1, len(condition_categories)))

        # For each condition category, check which conditions are present in the data.
        for med_category, conditions in condition_categories.items():
            present_conditions = [c for c in conditions if c in gender_data['target'].unique()]
            if present_conditions:
                conditions_order.extend(present_conditions)
                for condition in present_conditions:
                    condition_data = gender_data[gender_data['target'] == condition]
                    if not condition_data.empty:
                        best_performances[condition] = {}
                        best_models[condition] = {}
                        # For each model category, find the model (embedding) with the best average performance.
                        for model_cat, models in model_categories.items():
                            cat_performances = []
                            for model in models:
                                if model in condition_data['embedding'].unique():
                                    # Compute the mean performance (e.g. mean AUC) for this embedding.
                                    perf = condition_data[condition_data['embedding'] == model][metric].mean()
                                    # For AUC, clip values below 0.5 to 0.5.
                                    if metric == 'auc':
                                        perf = max(0.5, perf)
                                    cat_performances.append((perf, model))
                            if cat_performances:
                                best_perf, best_model = max(cat_performances, key=lambda x: x[0])
                                best_performances[condition][model_cat] = best_perf
                                best_models[condition][model_cat] = best_model

        if not best_performances:
            print(f"No best performance data for gender {gender}. Skipping radar plot.")
            continue

        # Create a polar plot (radar plot).
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='polar')

        n_conditions = len(conditions_order)
        # Compute evenly spaced angles around the circle.
        angles = [n / float(n_conditions) * 2 * np.pi for n in range(n_conditions)]
        angles += angles[:1]  # Close the circle

        # Choose a color for each model category.
        model_colors = plt.cm.Set2(np.linspace(0, 1, len(model_categories)))

        # Plot each model category's performance, with Baseline last (on top)
        model_cats = list(model_categories.keys())
        if 'Baseline' in model_cats:
            model_cats.remove('Baseline')
            model_cats.append('Baseline')
        for model_cat, color in zip(model_cats, model_colors):
            # For each condition (in the established order), get the best performance for this category.
            values = [best_performances[cond].get(model_cat, 0.5 if metric == 'auc' else 0) for cond in
                      conditions_order]
            values += values[:1]  # close the loop

            # Use thinner lines (reduced linewidth from 2 to 1)
            ax.plot(angles, values, '-', linewidth=1, label=model_cat, color=color)

            # Set different alpha (opacity) for Baseline vs other categories
            alpha = 0.9 if model_cat == 'Baseline' else 0.25
            ax.fill(angles, values, alpha=alpha, color=color)

        # Set the angles where the condition labels will appear.
        ax.set_xticks(angles[:-1])
        condition_colors = []
        prev_angle = 0
        for idx, (med_category, conditions) in enumerate(condition_categories.items()):
            cat_conditions = [c for c in conditions if c in conditions_order]
            if cat_conditions:
                angle_extent = (len(cat_conditions) / n_conditions) * 2 * np.pi
                mid_angle = prev_angle + angle_extent / 2
                # Add a separator line.
                ax.plot([prev_angle, prev_angle], [0.5 if metric == 'auc' else 0.4, 0.8], '--', color='gray', alpha=0.3)
                # Place the category label further out.
                label_radius = 0.9
                ax.text(mid_angle, label_radius, med_category,
                        ha='center', va='center',
                        rotation=0,
                        fontsize=10, fontweight='bold',
                        color=category_colors[idx])
                condition_colors.extend([category_colors[idx]] * len(cat_conditions))
                prev_angle += angle_extent

        # Remove default tick labels.
        ax.set_xticklabels([])

        # Add custom condition labels at the appropriate angles.
        for idx, (angle, condition) in enumerate(zip(angles[:-1], conditions_order)):
            label_radius = 0.8  # slightly outside the plotted data
            rotation = np.degrees(angle)
            # Rotate labels so they are horizontal.
            if angle >= np.pi / 2 and angle <= 3 * np.pi / 2:
                rotation += 180
            ax.text(angle, label_radius, condition,
                    rotation=rotation,
                    ha='center', va='center',
                    fontsize=8,
                    color=condition_colors[idx])

        # Set the radial limits and grid lines based on the metric.
        if metric == 'auc':
            # Updated y-limits from 0.5-0.9 to 0.5-0.8
            ax.set_ylim(0.5, 0.72)
            ax.set_rgrids([0.5, 0.55, 0.6, 0.65, 0.7], angle=0)
        else:
            ax.set_ylim(0.4, 0.9)
            ax.set_rgrids([0.5, 0.6, 0.7, 0.8], angle=0)

        plt.legend(loc='center left', bbox_to_anchor=(1.2, 1.0))
        # plt.tight_layout()
        plot_path = os.path.join(output_dir, f"radar_plot_best_performance_{gender}.pdf")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, format='pdf')
        plt.savefig(plot_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300, format='png')
        # plt.show()
        plt.close()

    return best_models


def compare_and_plot_combined_auc_across_seeds(prediction_results_dir, targets=None,
                                               output_dir=None, gender_specific=False,
                                               plot_radar=False,
                                               model_categories=None,
                                               condition_categories=None,
                                               include_ensembles=True):
    """
    For each target directory in prediction_results_dir:
      1. Reads prediction files (named like 'combined_predictions_embedding_EffNet_combined_91')
         that contain columns: index, true_values, predictions, fold.
      2. Aggregates the predictions (and true labels) across folds for each (seed, embedding) pair.
      3. Computes the combined AUC (using roc_auc_score) for each (seed, embedding).
      4. Performs paired Wilcoxon signedrank tests comparing each embeddings AUC to the baselines AUC.
      5. Computes prevalence (mean of all true_values * 100) for the target.
      6. Writes summary text files and creates boxplots via create_single_plot_new.
      7. If plot_radar is True, combines the pertarget summary data and calls create_radar_plots_from_combined_results.
    """
    # Create a timestamp for unique output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir is None:
        output_dir = os.path.join(prediction_results_dir, 'combined_auc_comparison')
    
    # Create a unique subdirectory with timestamp
    original_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    # Try to create the output directory, fall back to home directory if it fails
    try:
        os.makedirs(original_output_dir, exist_ok=True)
        output_dir = original_output_dir
        print(f"Saving results to: {output_dir}")
    except (PermissionError, OSError) as e:
        # If we can't create the directory, use a directory in the user's home folder
        home_dir = os.path.expanduser("~")
        backup_dir = os.path.join(home_dir, "deep_voice_plots", f"run_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"WARNING: Could not create directory at {original_output_dir}")
        print(f"Error: {str(e)}")
        print(f"Saving results to backup location: {backup_dir}")
        
        output_dir = backup_dir
    
    
    # Set default model and condition categories if not provided.
    if model_categories is None:
        model_categories = {
            'Baseline': ['baseline'],
            'MFCC': ['embedding_mfcc'],
            'Speaker Diarization': ['embedding_VoxCeleb', 'embedding_wavLMSD'],
            'Speaker Identification': ['embedding_xVector', 'embedding_EffNet', 'embedding_VoxCelebFintuned'],
            'Pretrained': ['embedding_wav2vec2Base', 'embedding_wav2vec2Large', 'embedding_wavLMBase',
                           'embedding_wavLMLarge', 'embedding_wav2vec2XLSRVanilla'],
            'Hebrew': ['embedding_wav2vec2XLSRVFineTuned', 'embedding_wav2vec2HebrewPretrained'],
            'Emotion Recognition': ['embedding_wavlmEM', 'embedding_wav2vecEM']
        }
        
        # Add Ensemble category if include_ensembles is True
        if include_ensembles:
            model_categories['Ensemble'] = []  # Will be populated with ensemble models later
    
    if condition_categories is None:
        condition_categories = {
            'Mental Health': ['Anxiety or fear-related disorders', 'Depressive disorders', 'Mood and anxiety disorders'],
            'Sleep': ['Sleep disorders', 'Obstructive sleep apnoea', 'has_SA'],
            'Respiratory': ['Respiratory disorders', 'Asthma', 'current_smoker', 'past_smoker'],
            'ENT': ['Rhinitis and sinusitis', 'Chronic rhinosinusitis'],
            'Neurological': ['Headache and migraine disorders', 'Migraine'],
            'Endocrine & Other': ['Thyroid disorders', 'hypothyroidism', 'COVID-19']
        }

    target_dirs = [d for d in os.listdir(prediction_results_dir)
                   if os.path.isdir(os.path.join(prediction_results_dir, d))]
    if gender_specific:
        target_dirs = [d for d in target_dirs if d.endswith('_male') or d.endswith('_female')]
    else:
        target_dirs = [d for d in target_dirs if d.endswith('_all')]
    if targets is not None:
        filtered_dirs = []
        for t in targets:
            filtered_dirs.extend([d for d in target_dirs if t in d])
        target_dirs = filtered_dirs
    target_dirs.sort()

    results_summary = {}  # to store summary results for each target
    combined_plot_data = []  # to accumulate data for radar plotting

    for target_dir in tqdm(target_dirs, desc="Processing Targets"):
        target_path = os.path.join(prediction_results_dir, target_dir)
        prediction_files = glob.glob(os.path.join(target_path, 'combined_predictions_*'))
        if not prediction_files:
            print(f"No prediction files found in {target_dir}. Skipping.")
            continue

        all_predictions = {}  # structure: { seed: { embedding: { 'true_values': [...], 'predictions': [...] } } }
        all_true_values = []

        for file in prediction_files:
            try:
                filename = os.path.basename(file)
                name_core = filename.replace('combined_predictions_', '').replace('.csv', '')
                parts = name_core.rsplit('_', 1)
                if len(parts) < 2:
                    print(f"File name {filename} does not conform to the expected pattern. Skipping.")
                    continue
                seed_str = parts[1]
                try:
                    seed = int(seed_str)
                except ValueError:
                    print(f"Could not parse seed from {filename}. Skipping.")
                    continue
                embedding_raw = parts[0]
                embedding = embedding_raw.replace('_combined', '')
                if embedding == 'baseline':
                    embedding = 'baseline'

                df = pd.read_csv(file)
                if 'true_values' not in df.columns or 'predictions' not in df.columns:
                    print(f"File {filename} does not have the expected columns. Skipping.")
                    continue

                all_true_values.extend(df['true_values'].tolist())
                if seed not in all_predictions:
                    all_predictions[seed] = {}
                if embedding not in all_predictions[seed]:
                    all_predictions[seed][embedding] = {'true_values': [], 'predictions': []}
                all_predictions[seed][embedding]['true_values'].extend(df['true_values'].tolist())
                all_predictions[seed][embedding]['predictions'].extend(df['predictions'].tolist())
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

        prevalence_val = np.mean(all_true_values) * 100 if len(all_true_values) > 0 else None

        auc_by_embedding = {}  # structure: { embedding: { seed: auc_value } }
        for seed, embeddings in all_predictions.items():
            for emb, data_dict in embeddings.items():
                y_true = np.array(data_dict['true_values'])
                y_pred = np.array(data_dict['predictions'])
                if len(np.unique(y_true)) < 2:
                    print(f"Seed {seed}, embedding {emb} in {target_dir}: Only one class present. Skipping AUC.")
                    continue
                try:
                    auc_val = roc_auc_score(y_true, y_pred)
                except Exception as e:
                    print(f"Error computing AUC for seed {seed}, embedding {emb} in {target_dir}: {e}")
                    continue
                if emb not in auc_by_embedding:
                    auc_by_embedding[emb] = {}
                auc_by_embedding[emb][seed] = auc_val

        wilcoxon_results = {}
        if 'baseline' not in auc_by_embedding:
            print(f"Baseline predictions not found in {target_dir}. Skipping Wilcoxon tests for this target.")
        else:
            baseline_seeds = set(auc_by_embedding['baseline'].keys())
            for emb, seed_auc in auc_by_embedding.items():
                if emb == 'baseline':
                    continue
                common_seeds = baseline_seeds.intersection(seed_auc.keys())
                if len(common_seeds) < 2:
                    print(f"Not enough common seeds for baseline and {emb} in {target_dir} (n={len(common_seeds)}).")
                    continue
                baseline_values = []
                emb_values = []
                for s in sorted(common_seeds):
                    baseline_values.append(auc_by_embedding['baseline'][s])
                    emb_values.append(seed_auc[s])
                try:
                    stat, p_value = wilcoxon(baseline_values, emb_values)
                except Exception as e:
                    print(f"Error performing Wilcoxon test for {emb} in {target_dir}: {e}")
                    stat, p_value = None, None
                wilcoxon_results[emb] = {
                    'n_seeds': len(common_seeds),
                    'baseline_median_auc': np.median(baseline_values),
                    'embedding_median_auc': np.median(emb_values),
                    'statistic': stat,
                    'p_value': p_value
                }

        results_summary[target_dir] = {
            'auc_by_embedding': auc_by_embedding,
            'wilcoxon': wilcoxon_results,
            'prevalence': prevalence_val
        }

        # Convert auc_by_embedding into a DataFrame for plotting.
        plot_rows = []
        for emb, seed_auc in auc_by_embedding.items():
            for s, auc_val in seed_auc.items():
                plot_rows.append({'embedding': emb, 'auc': auc_val, 'seed': s})
        if plot_rows:
            plot_df = pd.DataFrame(plot_rows)
            # Extract gender from target directory name if possible.
            gender_val = None
            parts = target_dir.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['male', 'female', 'all']:
                if parts[1] != 'all':
                    gender_val = parts[1]
            # Add columns for later radar plotting.
            target_name = target_dir.rsplit('_', 1)[0]
            plot_df['target'] = target_name
            plot_df['Gender'] = gender_val if gender_val is not None else 'all'
            combined_plot_data.append(plot_df)
            try:
                create_single_plot_new(plot_df, wilcoxon_results, target_dir, "auc",
                                       output_dir, gender=gender_val, prevalence=prevalence_val)
            except Exception as e:
                print(f"Error creating plot for target {target_dir}: {e}")

        # Write a summary file for this target.
        summary_file = os.path.join(output_dir, f"{target_dir}_auc_comparison.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Target: {target_dir}\n")
            f.write("Combined AUC by embedding (per seed):\n")
            for emb, seed_auc in auc_by_embedding.items():
                f.write(f"  {emb}:\n")
                for s, auc_val in sorted(seed_auc.items()):
                    f.write(f"    Seed {s}: {auc_val:.3f}\n")
            f.write("\nPaired Wilcoxon tests (comparing each embedding vs. baseline):\n")
            for emb, res in wilcoxon_results.items():
                f.write(
                    f"  {emb} vs baseline: n={res['n_seeds']}, baseline median AUC={res['baseline_median_auc']:.3f}, "
                    f"{emb} median AUC={res['embedding_median_auc']:.3f}, statistic={res['statistic']}, p-value={res['p_value']}\n"
                )
            if prevalence_val is not None:
                f.write(f"\nPrevalence: {prevalence_val:.1f}%\n")

    # If the flag is set, combine all per-target plotting data and create radar plots.
    if plot_radar and combined_plot_data:
        combined_data = pd.concat(combined_plot_data, ignore_index=True)
        
        # Find all ensemble models in the data and add them to the model_categories
        if include_ensembles:
            ensemble_models = [emb for emb in combined_data['embedding'].unique() 
                              if 'top' in str(emb)]
            if ensemble_models:
                model_categories['Ensemble'] = ensemble_models
                print(f"Added {len(ensemble_models)} ensemble models to radar plot: {ensemble_models}")
        
        # The combined_data DataFrame should now contain 'Gender', 'target', 'embedding', and 'auc'.
        create_radar_plots_from_combined_results(combined_data, model_categories, condition_categories, output_dir, metric='auc')

    # Write an overall summary file (across targets).
    overall_summary_file = os.path.join(output_dir, "overall_auc_comparison_summary.txt")
    with open(overall_summary_file, 'w') as f:
        for target, summary in results_summary.items():
            f.write(f"Target: {target}\n")
            auc_by_embedding = summary['auc_by_embedding']
            f.write("Combined AUC by embedding (per seed):\n")
            for emb, seed_auc in auc_by_embedding.items():
                f.write(f"  {emb}:\n")
                for s, auc_val in sorted(seed_auc.items()):
                    f.write(f"    Seed {s}: {auc_val:.3f}\n")
            f.write("Paired Wilcoxon tests (comparing to baseline):\n")
            for emb, res in summary['wilcoxon'].items():
                f.write(
                    f"  {emb} vs baseline: n={res['n_seeds']}, baseline median AUC={res['baseline_median_auc']:.3f}, "
                    f"{emb} median AUC={res['embedding_median_auc']:.3f}, statistic={res['statistic']}, p-value={res['p_value']}\n"
                )
            if summary['prevalence'] is not None:
                f.write(f"Prevalence: {summary['prevalence']:.1f}%\n")
            f.write("\n")

    print("Combined AUC computation, statistical tests, and plotting completed.")
    return results_summary



def get_full_target_name(target):
    """Get the full name of the target variable."""
    target_names = {
        "has_SA": "Sleep Apnea",
        "current_smoker": "Current Smoker",
        "past_smoker": "Past Smoker",
        "Cardiovascular disorders": "Cardiovascular Disorders",
        "Rhinitis and sinusitis": "Rhinitis and Sinusitis",
        "Obesity": "Obesity",
        "Asthma": "Asthma",
        "Mood and anxiety disorders": "Mood and Anxiety Disorders",
        "Vitamin deficiencies": "Vitamin Deficiencies",
        "Depressive disorders": "Depressive Disorders",
        "Episodic vestibular syndrome": "Episodic Vestibular Syndrome",
        "Anxiety or fear-related disorders": "Anxiety or Fear-Related Disorders"
    }
    return target_names.get(target, target)

if __name__ == "__main__":
    SEED = 37
    SUFFIX = "_lgb_HPO_first_seg_extraModels"
    data = f"/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/prediction_results/seed_{SEED}{SUFFIX}.csv"
    # plot_performance_metrics_no_combined(data, show_values=False, gender_specific=True, suffix=SUFFIX)
    # plot_performance_metrics(
    #     data, show_values=False, gender_specific=True, suffix=SUFFIX
    # )
    # plot_target_specific_aucs(data, target_names=["has_SA", "Cardiovascular disorders", "Rhinits and sinusitis",
    #                                               "past_smoker", "current_smoker", "obesity", "Asthma",
    #                                               "Mood and anxiety disorders", "Vitamin deficiency",
    #                                               "Depressive disorders", "Episodic vestibular syndrome",
    #                                               "Anxiety or fear-related disorders"],)

    #FIGS_DIR = f"/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/with_Pretrained/Filtered05_Nastya_Conditions/Figs"
    
    # Use os.path.expanduser to properly expand the tilde to your home directory
    import os
    home_dir = os.path.expanduser("~")
    pred_results_dir = os.path.join(home_dir, "prediction_results")
    FIGS_DIR = os.path.join(pred_results_dir, "figs")
    os.makedirs(FIGS_DIR, exist_ok=True)
    desired_targets =  [
                        'has_SA', 'COVID-19_male', 'Asthma'
                       ]
    # create_performance_plots_from_files(pred_results_dir, targets=None, output_dir=FIGS_DIR,
    #                                     generate_report=False, gender_specific=True, seed=SEED, create_radar=False)\

    compare_and_plot_combined_auc_across_seeds(pred_results_dir, targets=None, output_dir=FIGS_DIR, gender_specific=True,
                                               plot_radar=True, include_ensembles=True)
