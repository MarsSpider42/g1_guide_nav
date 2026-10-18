import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav_source"))

import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import lidar_processor
import snake_planner
import path_follower
import robot_marker
from command_view import CommandView


def show_occupancy_grid(p : lidar_processor.LidarProcessor):

        GRID_SIZE = p.occupancy_grid.shape[0]
        GRID_WIDTH_M = p.G_X_MAX-p.G_X_MIN
        GRID_HEIGHT_M = p.G_Y_MAX-p.G_Y_MIN

        fig, ax = plt.subplots(figsize=(8, 8))

        initial_grid = np.zeros((GRID_SIZE, GRID_SIZE))

        img = ax.imshow(
            initial_grid,
            origin='lower',
            cmap='gray',
            vmin=0,
            vmax=1,
            extent=[
                -GRID_WIDTH_M / 2,   # left
                GRID_WIDTH_M / 2,    # right
                0,                   # robot position
                GRID_HEIGHT_M        # forward distance
            ]
        )

        ax.set_title("Binary Occupancy Grid")

        ax.set_xlabel("Left / Right (m)")
        ax.set_ylabel("Forward Distance (m)")

        ax.set_aspect('equal')

        robot_marker.add_robot(ax)

        img.set_data(p.occupancy_grid)
        
        img.set_clim(0, 1)

        plt.show()

def show_cropped_cloud(p : lidar_processor.LidarProcessor, duration):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5
    )

    print("\nShowing cropped cloud...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Cropped LIDAR Cloud", width=900, height=600)
    vis.add_geometry(p.cropped_cloud)
    vis.add_geometry(frame)

    ctr = vis.get_view_control()
    # Smaller = further away
    # Larger = closer in
    ctr.set_zoom(0.3)
    ctr.set_front([0.4, -0.4, -0.6])   # viewing direction
    ctr.set_lookat([0, 0, 0])   # centre point
    ctr.set_up([0, 0, -1])      # up direction

    start_time = time.time()

    while True:
        vis.poll_events()
        vis.update_renderer()
        if time.time() - start_time > duration:
            break

    print("\nClosing window...")
    vis.destroy_window()

def show_std_reorient_cloud(p : lidar_processor.LidarProcessor, duration, side_on=False):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0
    )

    print("\nShowing cloud reoriented by -2.3 deg...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="-2.3 deg Reoriented LIDAR Cloud", width=1280, height=720)
    vis.add_geometry(p.std_reoriented_cloud)
    vis.add_geometry(frame)

    ctr = vis.get_view_control()
    # Smaller = further away
    # Larger = closer in
    ctr.set_zoom(0.3)
    if side_on:
        ctr.set_front([0, -1, 0])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, 0, -1])      # up direction
    else:
        ctr.set_front([0.4, -0.4, -0.6])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, 0, -1])      # up direction

    start_time = time.time()

    while True:
        vis.poll_events()
        vis.update_renderer()
        if time.time() - start_time > duration:
            break

    print("\nClosing window...")
    vis.destroy_window()

def show_ground_reorient_cloud(p : lidar_processor.LidarProcessor, duration, side_on=False):

    p.ground_plane_cloud.paint_uniform_color([0, 1, 0])

    ground_points = np.asarray(p.ground_plane_cloud.points)
    center = ground_points.mean(axis=0)
    normal_length = 0.75
    line_points = [
        center,
        center + p.best_normal * normal_length
    ]
    line_indices = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)

    # Red normal vector
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    # =========================================================
    # VISUALIZE PLANE FIT
    # =========================================================

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5
    )

    print("\nShowing detected plane...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detected Ground Plane", width=1280, height=720)
    vis.add_geometry(p.ground_plane_cloud)
    vis.add_geometry(p.non_ground)
    vis.add_geometry(line_set)
    vis.add_geometry(frame)

    ctr = vis.get_view_control()
    # Smaller = further away
    # Larger = closer in
    ctr.set_zoom(0.3)

    if side_on:
        ctr.set_front([0, -1, 0])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, 0, -1])      # up direction
    else:
        ctr.set_front([0.4, -0.4, -0.6])   # viewing direction
        ctr.set_lookat([0, 0, 0])   # centre point
        ctr.set_up([0, 0, -1])      # up direction

    start_time = time.time()

    while True:
        vis.poll_events()
        vis.update_renderer()
        if time.time() - start_time > duration:
            break

    print("\nClosing window...")

    vis.close()
    vis.destroy_window()

def show_feature_identified_cloud(p : lidar_processor.LidarProcessor, duration):
    
    # red ceil
    p.ceiling_cloud.paint_uniform_color([1,0,0])
    # green floor
    p.ground_plane_cloud.paint_uniform_color([0,1,0])
    # blue behind
    p.behind_cloud.paint_uniform_color([0,0,1])
    # Grey everything else
    p.obstacle_cloud.paint_uniform_color([0.6,0.6,0.6])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0
    )

    print("\nShowing ceiling, floor, and points behind robot...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Segmented LIDAR Cloud", width=1280, height=720)
    vis.add_geometry(p.ceiling_cloud)
    vis.add_geometry(p.ground_plane_cloud)
    vis.add_geometry(p.behind_cloud)
    vis.add_geometry(p.obstacle_cloud)
    vis.add_geometry(frame)

    ctr = vis.get_view_control()
    # Smaller = further away
    # Larger = closer in
    ctr.set_zoom(0.3)
    ctr.set_front([0.4, -0.4, -0.6])   # viewing direction
    ctr.set_lookat([0, 0, 0])   # centre point
    ctr.set_up([0, 0, -1])      # up direction

    start_time = time.time()

    while True:
        vis.poll_events()
        vis.update_renderer()
        if time.time() - start_time > duration:
            break

    print("\nClosing window...")
    vis.destroy_window()

def show_occupancy_grid_filtering_comparison(p : lidar_processor.LidarProcessor):
    
    rows, cols = p.raw_occupancy_grid.shape
    half_w = cols * p.G_RESOLUTION / 2
    extent = [-half_w, half_w, 0, rows * p.G_RESOLUTION]    # x: left/right (m), y: forward (m)

    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(p.raw_occupancy_grid, origin='lower', cmap='gray', extent=extent)
    ax1.set_title("Raw Occupancy Grid")
    ax1.set_xlabel("Left / Right (m)")
    ax1.set_ylabel("Forward Distance (m)")
    robot_marker.add_robot(ax1)                  # after imshow: it extends the bottom of the y-axis

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(p.occupancy_grid, origin='lower', cmap='gray', extent=extent)
    ax2.set_title("Cleaned Occupancy Grid")
    ax2.set_xlabel("Left / Right (m)")
    ax2.set_ylabel("Forward Distance (m)")
    robot_marker.add_robot(ax2)

    plt.show(block=True)

def show_height_grid(p : lidar_processor.LidarProcessor):
    rows, cols = p.raw_occupancy_grid.shape
    half_w = cols * p.G_RESOLUTION / 2
    extent = [-half_w, half_w, 0, rows * p.G_RESOLUTION]    # x: left/right (m), y: forward (m)

    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(1, 1, 1)
    im = ax1.imshow(p.height_grid, origin='lower', cmap='viridis', extent=extent)
    ax1.set_title("2.5D Coordinate-Relative Height Grid")
    ax1.set_xlabel("Left / Right (m)")
    ax1.set_ylabel("Forward Distance (m)")
    plt.colorbar(im, ax=ax1, label="Relative Height [m]") 
    robot_marker.add_robot(ax1)                  # after imshow: it extends the bottom of the y-axis

    plt.show(block=True)

def plot_snake(p : snake_planner.SnakePlanner, grid, rails):
    from matplotlib.lines import Line2D
    rows, cols = grid.shape
    half_w = cols * snake_planner.CELL_SIZE / 2
    extent = [-half_w, half_w, 0, rows * snake_planner.CELL_SIZE]    # x: left/right (m), y: forward (m)


    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(grid, cmap="gray", origin="lower", extent=extent)
    ax.set_autoscale_on(False)
    robot_marker.add_robot(ax)

    valid = [(r, v) for r, v in zip(rails, p.vertices) if r.valid]

    # free-space band per rail: a straight line for Straight_Rail, an arc for Radial_Rail
    for r, _ in valid:
        xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
        ax.plot(xs, ys, color="cyan", linewidth=0.5, alpha=0.5)

    for r in rails:
        is_open = not (r.left_closed and r.right_closed)
        if not is_open and not r.valid:
            continue                              # no free space: nothing to draw
        xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
        if is_open:
            ax.plot(xs, ys, color="orange", linewidth=1.0, alpha=0.8,
                    linestyle="-" if r.valid else "--")
        else:
            ax.plot(xs, ys, color="cyan", linewidth=0.5, alpha=0.5)

    # snake vertices (valid rails only)
    pts = [r.to_xy(v) for r, v in valid]
    ax.plot([float(p[0]) for p in pts], [float(p[1]) for p in pts],
            color="lime", linewidth=2, marker="o", markersize=3)

    handles = [Line2D([0], [0], color="cyan", lw=1, label="Closed rail"),
               Line2D([0], [0], color="orange", lw=1, label="Open rail (used)"),
               Line2D([0], [0], color="orange", lw=1, ls="--", label="Open rail (dropped, corridor)"),
               Line2D([0], [0], color="lime", lw=2, marker="o", markersize=3, label="Snake")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.set_title("Path Snake, 18 Rails spaced 1-3.5m from G1")
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")
    plt.show()

def plot_blindspot_mask(p : lidar_processor.LidarProcessor, show_fill=True, comparison=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    raw  = p.raw_occupancy_grid
    filt = p.occupancy_grid
    mask = p.blindspot_m
    res  = p.G_RESOLUTION

    # grid rows = forward (G_X range), columns = left/right (G_Y range), robot at row 0
    extent = [p.G_Y_MIN, p.G_Y_MAX, p.G_X_MIN, p.G_X_MAX]
    H, W = mask.shape
    lateral = p.G_Y_MIN + (np.arange(W) + 0.5) * res   # cell-centre coordinates,
    forward = p.G_X_MIN + (np.arange(H) + 0.5) * res   # same convention the mask uses

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    panels = [(raw, "(a) Raw occupancy grid"), (filt, "(b) After blind-spot closing & filtering")]

    for ax, (grid, title) in zip(axes, panels):
        # occupied = black, free = white
        ax.imshow(grid, cmap="gray_r", origin="lower", extent=extent,
                  vmin=0, vmax=1, interpolation="nearest")

        if show_fill:   # light tint inside the mask
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", origin="lower",
                      extent=extent, alpha=0.25, interpolation="nearest")

        # mask outline: the 0.5 contour of the boolean mask traces its edge
        ax.contour(lateral, forward, mask.astype(float), levels=[0.5],
                   colors="tab:orange", linewidths=1.5)

        # robot footprint (approx. 0.45 m wide x 0.2 m deep), centred at the origin
        robot_marker.add_robot(ax)

        ax.set_title(title)
        ax.set_xlabel("Left / Right (m)")
        ax.set_aspect("equal")

    axes[0].set_ylabel("Forward distance (m)")

    handles = [Line2D([0], [0], color="tab:orange", lw=1.5, label="Blind-spot mask edge"),
               Patch(color="tab:red", label="Robot")]
    if show_fill:
        handles.insert(1, Patch(color="orange", alpha=0.25, label="Masked region"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()

def plot_guidance_point(p1 : snake_planner.SnakePlanner, p2 : path_follower.PathFollower, grid, rails, show_planner=False):
    from matplotlib.lines import Line2D
    rows, cols = grid.shape
    half_w = cols * snake_planner.CELL_SIZE / 2
    extent = [-half_w, half_w, 0, rows * snake_planner.CELL_SIZE]    # x: left/right (m), y: forward (m)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(grid, cmap="gray", origin="lower", extent=extent)
    ax.set_autoscale_on(False)
    robot_marker.add_robot(ax)

    handles = []

    if show_planner:
        # rails: same colour scheme as plot_snake
        for r in rails:
            is_open = not (r.left_closed and r.right_closed)
            if not is_open and not r.valid:
                continue                              # no free space: nothing to draw
            xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
            if is_open:
                ax.plot(xs, ys, color="orange", linewidth=1.0, alpha=0.8,
                        linestyle="-" if r.valid else "--")
            else:
                ax.plot(xs, ys, color="cyan", linewidth=1.0, alpha=0.8)

        # snake through valid vertices
        pts = [r.to_xy(v) for r, v in zip(rails, p1.vertices) if r.valid]
        ax.plot([float(pt[0]) for pt in pts], [float(pt[1]) for pt in pts],
                color="lime", linewidth=2, marker="o", markersize=3)

        handles += [Line2D([0], [0], color="cyan", lw=1, label="Closed rail"),
                    Line2D([0], [0], color="orange", lw=1, label="Open rail (used)"),
                    Line2D([0], [0], color="orange", lw=1, ls="--", label="Open rail (dropped)"),
                    Line2D([0], [0], color="lime", lw=2, marker="o", markersize=3, label="Snake")]

    # guidance point: PathFollower stores it as (right, forward) in metres, or None if it stopped
    if p2.target_xy is not None:
        gx, gy = p2.target_xy
        ax.plot([0, gx], [0, gy], color="yellow", linewidth=1.5, linestyle=":")   # straight path that was checked
        ax.plot(gx, gy, marker="*", color="yellow", markersize=20,
                markeredgecolor="black", zorder=4)
        handles += [Line2D([0], [0], color="yellow", marker="*", markersize=12, lw=0,
                           markeredgecolor="black", label="Guidance point"),
                    Line2D([0], [0], color="yellow", lw=1.5, ls=":", label="Checked path")]

    ax.set_title("Guidance Point" if p2.target_xy is not None else "No valid guidance point")
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")

    plt.show()

def mode_of(cmd):
    if cmd.status != "ok":
        return "stop"
    if cmd.vx > 0 and cmd.omega != 0:
        return "arc"
    if cmd.vx > 0:
        return "drive"
    return "pivot" if cmd.omega != 0 else "stop"


def set_arrow(arrow, start, end):
    """Show a FancyArrowPatch between two points, or hide it if there is nothing to draw."""
    if end is None or np.hypot(end[0] - start[0], end[1] - start[1]) < 1e-3:
        arrow.set_visible(False)
    else:
        arrow.set_positions(start, end)
        arrow.set_visible(True)

MODE_COLORS = {"drive": "lime", "arc": "orange", "pivot": "red", "stop": "lightgrey"}

def plot_command_signal(p : path_follower.PathFollower, reach_m=2.5):
    from matplotlib.patches import FancyArrowPatch
    cmd, tgt, err = p.last_cmd, p.target_xy, p.err_f
    if cmd is None:
        raise ValueError("Run p.compute(...) before plotting")
    max_vx, max_omega = p.config.max_vx, p.config.max_omega

    fig, ax = plt.subplots(figsize=(8, 8))
    robot_marker.add_robot(ax)                                  # call first: it adjusts the y limits
    ax.set_facecolor("black")
    ax.set_xlim(-reach_m, reach_m)
    ax.set_ylim(0, reach_m)
    ax.set_aspect("equal")
    ax.grid(True, color="0.25", linewidth=0.5)
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")
    ax.set_title("Generated Locomotion Command")

    def arrow(color, width):
        a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=22,
                            color=color, linewidth=width, zorder=4)
        a.set_visible(False)
        ax.add_patch(a)
        return a

    # robot -> guidance point
    set_arrow(arrow("yellow", 2.5), (0.0, 0.0), tgt)

    # forward speed
    front = robot_marker.ROBOT_DEPTH_M / 2
    set_arrow(arrow("lime", 5), (0.0, front),
              (0.0, front + cmd.vx / max_vx) if cmd.vx > 0 else None)

    # turn arc: sweep grows with |omega|, + omega = left = counter-clockwise
    if cmd.omega != 0:
        sweep = np.radians(60) * min(1.0, abs(cmd.omega) / max_omega)
        phi = np.pi / 2 + np.sign(cmd.omega) * np.linspace(0.0, sweep, 30)
        xs, ys = 0.5 * np.cos(phi), 0.5 * np.sin(phi)
        ax.plot(xs, ys, color="orange", linewidth=3, zorder=4)
        set_arrow(arrow("orange", 3), (xs[-3], ys[-3]), (xs[-1], ys[-1]))

    # command readout
    m = mode_of(cmd)
    turn = "left" if cmd.omega > 0 else "right" if cmd.omega < 0 else "-"
    err_txt = "-" if err is None else f"{np.degrees(err):+.1f} deg"
    lines = [m.upper(), f"vx    {cmd.vx:+.2f} m/s", f"omega {cmd.omega:+.2f} rad/s ({turn})",
             f"error {err_txt}"]
    if cmd.status != "ok":
        lines.append(cmd.status)
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=14, family="monospace", color=MODE_COLORS[m],
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.75, edgecolor="white"))

    plt.show()

def show_command_view_frame(grid, rails, planner, follower,
                            frame=None, timing=None, reach_m=2.0):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    """Show the two-pane CommandView for a single frame and keep the window open.
    Call after:
        planner.plan(rails)
        cmd = follower.compute(rails, planner.vertices, grid)
    timing (optional): dict like {"ground_ms": 12.3, "snake_ms": 2.1}; keys must end in "_ms"."""
    view = CommandView(num_rails=len(rails), reach_m=reach_m, grid_shape=grid.shape,
                       max_vx=follower.config.max_vx, max_omega=follower.config.max_omega)

    view.update({
        "grid": grid,
        "rails": rails,
        "vertices": planner.vertices,
        "cmd": follower.last_cmd,
        "target_xy": follower.target_xy,
        "err": follower.err_f,          # smoothed heading error, as the live view uses
        "frame": "-" if frame is None else frame,
        "timing": timing or {},
    })

    # CommandView turns interactive mode on for live playback; switch it off
    # so plt.show() blocks and the window stays open
    plt.ioff()
    plt.show()


def _draw_guidance_command(ax, grid, rails, vertices, target_xy, cmd, title,
                           show_planner=True, crop=(-2.0, 2.0, 0.0, 3.5), max_vx=0.4, max_omega=0.7):
    from matplotlib.patches import FancyArrowPatch
    """One panel: grid (+ rails/snake), guidance point, and the command drawn at the robot."""
    rows, cols = grid.shape
    half_w = cols * snake_planner.CELL_SIZE / 2
    ax.imshow(grid, cmap="gray", origin="lower", extent=[-half_w, half_w, 0, rows * snake_planner.CELL_SIZE])
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[2] - 0.1, crop[3])
    ax.set_autoscale_on(False)
    ax.set_aspect("equal")

    if show_planner:
        for r in rails:
            is_open = not (r.left_closed and r.right_closed)
            if not is_open and not r.valid:
                continue
            xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
            ax.plot(xs, ys, color="orange" if is_open else "cyan", linewidth=0.8, alpha=0.7,
                    linestyle="--" if (is_open and not r.valid) else "-")
        pts = [r.to_xy(v) for r, v in zip(rails, vertices) if r.valid]
        ax.plot([float(p[0]) for p in pts], [float(p[1]) for p in pts],
                color="lime", linewidth=1.5, marker="o", markersize=2, alpha=0.8)

    # guidance point and the straight path the follower checked
    if target_xy is not None:
        gx, gy = target_xy
        ax.plot([0, gx], [0, gy], color="yellow", linewidth=1.2, linestyle=":")
        ax.plot(gx, gy, marker="*", color="yellow", markersize=16, markeredgecolor="black", zorder=5)

    robot_marker.add_robot(ax)

    # command glyph at the robot: forward arrow (vx) and turn arc (omega)
    front = robot_marker.ROBOT_DEPTH_M
    if cmd.vx > 0:
        ax.add_patch(FancyArrowPatch((0, front), (0, front + 0.6 * cmd.vx / max_vx), arrowstyle="-|>",
                                     mutation_scale=14, color="lime", linewidth=3, zorder=6))
    if cmd.omega != 0:
        sweep = np.radians(70) * min(1.0, abs(cmd.omega) / max_omega)
        phi = np.pi / 2 + np.sign(cmd.omega) * np.linspace(0.0, sweep, 30)   # + omega = left
        xs, ys = 0.35 * np.cos(phi), 0.35 * np.sin(phi)
        ax.plot(xs, ys, color="orange", linewidth=2.5, zorder=6)
        ax.add_patch(FancyArrowPatch((xs[-3], ys[-3]), (xs[-1], ys[-1]), arrowstyle="-|>",
                                     mutation_scale=14, color="orange", linewidth=2.5, zorder=6))

    # mode label
    mode = mode_of(cmd)
    ax.text(0.03, 0.97, f"{mode.upper()}\nvx {cmd.vx:.1f} m/s\nω {cmd.omega:+.1f} rad/s",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace",
            color=MODE_COLORS[mode], bbox=dict(boxstyle="round", facecolor="black", alpha=0.75))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_command_panels(cases, ncols=4, show_planner=True, save_path=None):
    """cases: list of dicts from NavPipeline.process (grid, rails, vertices, target_xy, cmd) plus 'title'."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cases):
        _draw_guidance_command(ax, c["grid"], c["rails"], c["vertices"], c["target_xy"], c["cmd"],
                               c["title"], show_planner=show_planner)
    for ax in list(axes.flat)[len(cases):]:
        ax.axis("off")

    handles = [Line2D([0], [0], color="yellow", marker="*", markersize=10, lw=0,
                      markeredgecolor="black", label="Guidance point"),
               Line2D([0], [0], color="yellow", lw=2.5, ls=":", label="Checked path",
                      path_effects=[pe.Stroke(linewidth=3.5, foreground="black"), pe.Normal()]),
               Line2D([0], [0], color="lime", lw=3, label="Forward command"),
               Line2D([0], [0], color="orange", lw=2.5, label="Turn command")]
    if show_planner:
        handles = [Line2D([0], [0], color="cyan", lw=1, label="Closed rail"),
                   Line2D([0], [0], color="orange", lw=1, alpha=0.7, label="Open rail"),
                   Line2D([0], [0], color="lime", lw=1.5, marker="o", markersize=2, label="Snake")] + handles
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()