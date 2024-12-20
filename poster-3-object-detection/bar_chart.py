from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Define colors and other parameters
color_primary = '#990000'  # University red
color_secondary = '#2F3EEA'  # University blue
color_tertiary = '#F6D04D'  # University gold

# Update matplotlib parameters for global styling
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'text.usetex': True})
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 42

def plot_experiment_results(experiments, map_values):
    """
    Plots a square-shaped bar chart of experiments vs. mAP values.

    Parameters:
    - experiments: List of strings containing experiment names.
    - map_values: List of floats containing mAP values corresponding to the experiments.
    """
    # Ensure the lists have the same length
    assert len(experiments) == len(map_values), "Experiments and mAP values lists must be of the same length."

    # Define a fixed set of colors
    base_colors = [
        "#990000", "#2F3EEA", "#1FD082", "#030F4F", "#F6D04D", 
        "#FC7634", "#F7BBB1", "#DADADA", "#E83F48", "#008835", "#79238E"
    ]

    # If more experiments than colors, generate additional colors dynamically
    if len(experiments) > len(base_colors):
        additional_colors = cm.tab20(np.linspace(0, 1, len(experiments) - len(base_colors)))
        # Convert RGBA from Matplotlib colormap to Hex
        additional_colors = [mpl.colors.rgb2hex(c) for c in additional_colors]
        all_colors = base_colors + additional_colors
    else:
        all_colors = base_colors

    # Assign colors to experiments
    bar_colors = [all_colors[i % len(all_colors)] for i in range(len(experiments))]

    # Update matplotlib parameters for global styling
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        'text.usetex': False,  # Disable LaTeX for easier compatibility
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 42
    })

    # Create a square figure
    plt.figure(figsize=(8, 8))  # Square shape: equal width and height

    # Positions of the bars on the x-axis
    indices = np.arange(len(experiments))

    # Plot the bars with transparency and black edge
    bars = plt.bar(
        indices, 
        map_values, 
        color=bar_colors, 
        alpha=0.7,  # Set transparency
        edgecolor='black'  # Black border around bars
    )

    # Add labels and title
    plt.xlabel('Experiments')
    plt.ylabel('mAP Values')
    plt.title('Experiment mAP Comparison', color=color_primary)

    # Set the position and labels of the x-ticks
    plt.xticks(indices, experiments, rotation=45, ha='right')

    # Add mAP value labels on top of each bar
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # Slightly above the bar
            f'{map_values[idx]:.2f}',
            ha='center',
            va='bottom',
            fontsize=20, 
        )

    # Adjust layout to prevent clipping of tick labels
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig('figures/svg/experiment_map_comparison.svg', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

# Example usage
list_of_experiments = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5', 'experiment_6', 'experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5', 'experiment_6']
map_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

plot_experiment_results(list_of_experiments, map_values)

print("Successfully plotted")
