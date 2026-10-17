from matplotlib.patches import Rectangle

ROBOT_WIDTH_M = 0.45    # along x (left/right)
ROBOT_DEPTH_M = 0.2/2    # along y (forward)


def add_robot(ax, color="red"):
    """Robot as a rectangle centred on the sensor origin, long edge along x."""
    rect = Rectangle((-ROBOT_WIDTH_M / 2, 0), ROBOT_WIDTH_M, ROBOT_DEPTH_M,
                     facecolor=color, edgecolor="white", linewidth=1, zorder=5)
    ax.add_patch(rect)
    return rect