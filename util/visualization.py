import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


color_list = ['blue', 'red', 'green']  # Colors for the labels
water_label_dict = {0:'empty', 1:'H', 2:'O'}
Sb2S3_label_dict = {0:'empty', 1:'Sb', 2:'S'}


def plot_3d_grid(points, labels, num_classes=2, title="3D Grid Points Colored by Label"):
    """
    Plots a 3D grid of points colored by their labels.
    
    Parameters:
    - points: A 3D numpy array of shape (3, N, N, N) representing the coordinates.
    - labels: A 4D numpy array of shape (1, N, N, N) representing the labels for each point.
    """
    # Reshape to list of 3D points
    x, y, z = points
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    labels = labels.flatten()

    # Split points by label
    points_labels = []
    for i in range(num_classes):
        points_label = (x[labels == i], y[labels == i], z[labels == i])
        points_labels.append(points_label)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_classes):
        ax.scatter(*points_labels[i], c=color_list[i], label=water_label_dict[i], alpha=0.5, s=1)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()


def visualize_voxel_labels(points, num_classes=3, title="3D Grid Points Colored by Label"):
    """
    Visualizes a 3D voxel grid with points colored by their labels.
    
    Parameters:
    - points: A 3D numpy array of shape (N, N, N) representing the voxel.
    - num_classes: Number of classes for coloring the points.
    - title: Title for the plot.
    """
    # Get voxel coordinates
    x, y, z = np.indices(points.shape)

    # Flatten
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    labels = points.flatten()

    # Split points by label
    points_labels = []
    for i in range(1, num_classes):
        points_label = (x[labels == i], y[labels == i], z[labels == i])
        points_labels.append(points_label)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_classes - 1):
        ax.scatter(*points_labels[i], c=color_list[i], label=water_label_dict[i+1], alpha=0.5, s=10)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"samples/{title}.png", dpi=300)
    plt.close()


def visualize_dataset(num_samples=5):
    from utils import get_dataloaders
    from config.config import _C as cfg

    tr, te, dataDims = get_dataloaders(cfg)
    for i in range(num_samples):
        visualize_voxel_labels(tr.dataset[i].squeeze()*3-1, num_classes=3, title=f"train_voxel_{i}")



if __name__ == "__main__":
    # points = np.load("samples/run_ep5_samples40.npy")  # Load your points data
    # points = points.squeeze()
    # visualize_voxel_labels(points, num_classes=3, title="water_voxel")
    visualize_dataset()