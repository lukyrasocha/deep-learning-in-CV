import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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


df = pd.read_csv('figures/number_of_proposals.csv')

iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

# Prepare the figure
plt.figure(figsize=(12, 6))

# Store handles and labels for manual legend
handles = []
labels = []

# Plot each IoU threshold
for i, iou_thresh in enumerate(iou_thresholds):
    subset = df[df['iou_threshold'] == iou_thresh]
    line, = plt.plot(subset['max_proposals'], subset['avg_recall'], marker='o',
                     label=f'IoU Threshold: {iou_thresh}')
    handles.append(line)  # Collect the line for the legend
    labels.append(fr'IoU $>$ {iou_thresh} Pothole, IoU {iou_thresh} $\leq$ Background')  # Custom label

# Add labels and title
plt.xlabel('Number of Proposals')
plt.ylabel('Average Recall')
plt.title('Average Recall vs Number of Proposals for Different IoU Thresholds', color=color_primary)

# Add manual legend
plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)

# Add grid and save the plot
plt.grid(True)
plt.savefig('figures/svg/avr_recall_vs_num_proposals.svg', dpi=300, bbox_inches='tight')
plt.show()


# Prepare the figure
plt.figure(figsize=(12, 6))

# Store handles and labels for manual legend
handles = []
labels = []

# Plot each IoU threshold
for i, iou_thresh in enumerate(iou_thresholds):
    subset = df[df['iou_threshold'] == iou_thresh]
    line, = plt.plot(subset['max_proposals'], subset['avg_mabo'], marker='o',
                     label=f'IoU Threshold: {iou_thresh}')
    handles.append(line)  # Collect the line for the legend
    labels.append(fr'IoU $>$ {iou_thresh} Pothole, IoU {iou_thresh} $\leq$ Background')  # Custom label

# Add labels and title
plt.xlabel('Number of Proposals')
plt.ylabel('Average Mabo')
plt.title('Average Recall vs Number of Proposals for Different IoU Thresholds', color=color_primary)

# add legend
plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)

# set limits and grid
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('figures/svg/avr_mabo_vs_num_proposals.svg', dpi=300, bbox_inches='tight')
plt.show()

print("Suffessfully plotted the recall and mabo vs number of proposals for different IoU thresholds")
