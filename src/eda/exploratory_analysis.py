import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
from src.preprocessing.constants import ACTIVITY_MAP

# Six activities chosen to plot sensor data over time:
# lying --> near-zero movement
# sitting --> slight hand movement
# walking --> moderate intensity
# running --> high intensity
# ascending stairs --> intermittent vertical movement
# rope jumping --> high intensity and high-amplitude movement
SNIPPET_ACTIVITIES = [1, 2, 4, 5, 12, 24]

def plot_activity_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df['activity_id'].value_counts().sort_index()
    labels = [ACTIVITY_MAP.get(i, f"act_{i}") for i in counts.index]
    plt.figure(figsize=(14, 6))
    sns.barplot(x=labels, y=counts.values, hue=labels, palette='viridis', legend=False)
    plt.title('Activity Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'activity_distribution.png', dpi=150)
    plt.close()

def plot_sensor_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    sensor_cols = [c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")]
    corr = df[sensor_cols].corr()
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Sensor Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_correlation.png', dpi=150)
    plt.close()

def plot_pca_clusters(df: pd.DataFrame, output_dir: Path, n_sample: int = 50_000) -> None:
    sensor_cols = [c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")]
    df_sub = df.sample(n=min(n_sample, len(df)), random_state=42).copy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_sub[sensor_cols])
    df_sub['pca_1'], df_sub['pca_2'] = coords[:, 0], coords[:, 1]
    df_sub['activity_name'] = df_sub['activity_id'].map(ACTIVITY_MAP)
    
    plt.figure(figsize=(11, 9))
    sns.scatterplot(data=df_sub, x='pca_1', y='pca_2', hue='activity_name', alpha=0.45, s=12, linewidth=0)
    plt.title('PCA Activity Clusters')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_clusters.png', dpi=150)
    plt.close()

def plot_pca_variance(df: pd.DataFrame, output_dir: Path) -> None:
    """Enhanced PCA plot: Scree plot of explained variance to guide feature selection."""
    sensor_cols = [c for c in df.columns if c not in ("timestamp", "activity_id", "subject_id")]
    pca = PCA().fit(df[sensor_cols])
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.7, align='center', label='Individual')
    plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', label='Cumulative')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.title('PCA Explained Variance (Scree Plot)')

    # Top features for PC1
    pc1_loadings = pd.Series(pca.components_[0], index=sensor_cols).abs().sort_values(ascending=False).head(10)
    plt.subplot(1, 2, 2)
    pc1_loadings.plot(kind='barh', color='teal')
    plt.title('Top 10 Feature Contributions to PC1')
    plt.xlabel('Absolute Loading Coefficient')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_variance_guide.png', dpi=150)
    plt.close()

def plot_sensor_snippets(df: pd.DataFrame, output_dir: Path) -> None:
    present = set(df['activity_id'].unique())
    acts = [a for a in SNIPPET_ACTIVITIES if a in present]
    n = len(acts)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n))
    if n == 1: axes = [axes]
    for ax, act_id in zip(axes, acts):
        data = df[df['activity_id'] == act_id]
        samp = data.iloc[1000 : 2000] if len(data) > 2000 else data.head(1000)
        if samp.empty: continue
        t = np.arange(len(samp)) / 100.0
        ax.plot(t, samp['hand_acc16_x'], label='Hand X')
        ax.plot(t, samp['chest_acc16_x'], label='Chest X')
        ax.plot(t, samp['ankle_acc16_x'], label='Ankle X')
        ax.set_title(f'Activity: {ACTIVITY_MAP.get(act_id, act_id)}')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_snippets.png', dpi=150)
    plt.close()

def run_eda(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_activity_distribution(df, output_dir)
    plot_sensor_correlation(df, output_dir)
    plot_pca_clusters(df, output_dir)
    plot_pca_variance(df, output_dir)
    plot_sensor_snippets(df, output_dir)
    print(f" EDA plots saved to {output_dir}")

