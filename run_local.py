import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav_source"))

import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import bag_list
import lidar_processor
import snake_planner
import path_follower
import local_vis_wrapper
from nav_pipeline import NavPipeline
from command_view import CommandView
from datasets.lidar_scan_storage import ScanStore, scan_to_xyz

WINDOW_DURATION = 5 #s

def pull_single_scan(path, index, debug):
    bag = ScanStore(path)

    if debug:
        print(f"\nBag at path: {path}, # Scans in bag: {len(bag)}, \nMetadata: {bag.meta}")

    frame = bag[index]

    if debug:
        print(f"\nFrame: {index}, Frame shape: {np.shape(frame)}, # Points in scan: {bag.point_num(index)}, Frame timestamp: {bag.timestamp(index)} \nSample: {frame[0:5]}")

    array = scan_to_xyz(frame)

    if debug:
        print(f"\nPoint array size: {np.shape(array)}, Type: {type(array)}\nSample: {array[0:5,:]}")

    return array

def run_live_occupancy_grid(bag_path, duration=150, start_frame=0, debug=False):

    # =====================================================
    # OCCUPANCY GRID SETTINGS
    # =====================================================
    L = lidar_processor.LidarProcessor(None)
    GRID_SIZE = 80
    GRID_WIDTH_M = L.G_X_MAX - L.G_X_MIN
    GRID_HEIGHT_M = L.G_Y_MAX - L.G_Y_MIN
    del L

    # =====================================================
    # MATPLOTLIB INTERACTIVE SETUP
    # =====================================================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    initial_grid = np.zeros((GRID_SIZE, GRID_SIZE))

    img = ax.imshow(
        initial_grid,
        origin='lower',
        cmap='gray',
        vmin=0,
        vmax=1,
        extent=[
            -GRID_WIDTH_M / 2,
            GRID_WIDTH_M / 2,
            0,
            GRID_HEIGHT_M
        ]
    )

    ax.arrow(
        0, 0, 0.0, 0.4,
        width=GRID_WIDTH_M * 0.01,
        head_width=GRID_WIDTH_M * 0.03,
        head_length=GRID_HEIGHT_M * 0.03,
        length_includes_head=True,
        color='red'
    )

    ax.set_title("Live Binary Occupancy Grid")
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")
    ax.set_aspect('equal')
    plt.show(block=False)

    # =====================================================
    # PLAYBACK CONTROL
    # =====================================================
    paused = False

    def on_key(event):
        nonlocal paused
        print(f"Key pressed: {event.key}")
        if event.key == 'space':
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # =====================================================
    # PROCESS SCANS
    # =====================================================
    num_scans = len(ScanStore(bag_path))   # just to know how many frames exist

    LP = lidar_processor.LidarProcessor(None)
    start_time = time.time()

    for i in range(start_frame, num_scans):

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            print(f"Reached duration limit: {elapsed_time:.2f}s")
            break

        while paused:
            plt.pause(0.05)

        # -------------------------------------------------
        # LOAD SCAN
        # -------------------------------------------------
        points = pull_single_scan(bag_path, i, debug)

        # -------------------------------------------------
        # PROCESSING PIPELINE
        # -------------------------------------------------
        LP.update_cloud(points)
        LP.process(False, False)

        # -------------------------------------------------
        # OCCUPANCY GRID UPDATE
        # -------------------------------------------------
        img.set_data(LP.occupancy_grid)
        img.set_clim(0, 1)
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if (i % 5 == 0) & (debug):
            print(f"Frame {i}")
        time.sleep(0.01)

    plt.ioff()
    plt.show()

def run_live_height_grid(bag_path, duration=150, start_frame=0, debug=False):

    # =====================================================
    # OCCUPANCY GRID SETTINGS
    # =====================================================
    L = lidar_processor.LidarProcessor(None)
    GRID_SIZE = 80
    GRID_WIDTH_M = L.G_X_MAX - L.G_X_MIN
    GRID_HEIGHT_M = L.G_Y_MAX - L.G_Y_MIN
    del L

    # =====================================================
    # MATPLOTLIB INTERACTIVE SETUP
    # =====================================================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    initial_grid = np.zeros((GRID_SIZE, GRID_SIZE))

    img = ax.imshow(
        initial_grid,
        origin='lower',
        cmap='coolwarm',
        vmin=0,
        vmax=2.0,
        extent=[
            -GRID_WIDTH_M / 2,
            GRID_WIDTH_M / 2,
            0,
            GRID_HEIGHT_M
        ]
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Height Above Ground (m)")

    ax.arrow(
        0, 0, 0.0, 0.4,
        width=GRID_WIDTH_M * 0.01,
        head_width=GRID_WIDTH_M * 0.03,
        head_length=GRID_HEIGHT_M * 0.03,
        length_includes_head=True,
        color='red'
    )

    ax.set_title("Live 2.5D Height Grid")
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")
    ax.set_aspect('equal')
    plt.show(block=False)

    # =====================================================
    # PLAYBACK CONTROL
    # =====================================================
    paused = False

    def on_key(event):
        nonlocal paused
        print(f"Key pressed: {event.key}")
        if event.key == 'space':
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # =====================================================
    # PROCESS SCANS
    # =====================================================
    num_scans = len(ScanStore(bag_path))

    LP = lidar_processor.LidarProcessor(None)
    start_time = time.time()

    for i in range(start_frame, num_scans):

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            print(f"Reached duration limit: {elapsed_time:.2f}s")
            break

        while paused:
            plt.pause(0.05)

        # -------------------------------------------------
        # LOAD SCAN
        # -------------------------------------------------
        points = pull_single_scan(bag_path, i, debug)

        # -------------------------------------------------
        # PROCESSING PIPELINE
        # -------------------------------------------------
        LP.update_cloud(points)
        LP.process(False, False)

        # -------------------------------------------------
        # HEIGHT GRID UPDATE
        # -------------------------------------------------
        display_grid = np.nan_to_num(LP.height_grid)

        img.set_data(display_grid)
        img.set_clim(0, max(0.5, np.nanmax(display_grid)))

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if (i % 5 == 0) & (debug):
            print(f"Frame {i}")

        time.sleep(0.01)

    plt.ioff()
    plt.show()

def run_live_cloud_view(bag_path, duration=150, start_frame=0, debug=False):

    # =====================================================
    # OPEN3D PERSISTENT VISUALISER
    # =====================================================
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720, window_name="LiDAR Segmentation")

    floor_cloud = o3d.geometry.PointCloud()
    ceil_cloud = o3d.geometry.PointCloud()
    behind_cloud = o3d.geometry.PointCloud()
    other_cloud = o3d.geometry.PointCloud()

    vis.add_geometry(floor_cloud)
    vis.add_geometry(ceil_cloud)
    vis.add_geometry(behind_cloud)
    vis.add_geometry(other_cloud)

    frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(frame_axes)

    # =====================================================
    # PLAYBACK CONTROL
    # =====================================================
    state = {"paused": False}

    def toggle_pause(vis_obj):
        state["paused"] = not state["paused"]
        print("PAUSED" if state["paused"] else "RESUMED")
        return False

    vis.register_key_callback(ord(" "), toggle_pause)

    # =====================================================
    # PROCESS SCANS
    # =====================================================
    num_scans = len(ScanStore(bag_path))

    LP = lidar_processor.LidarProcessor(None)
    start_time = time.time()

    for i in range(start_frame, num_scans):

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            print(f"Reached duration limit: {elapsed_time:.2f}s")
            break

        while state["paused"]:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)

        # -------------------------------------------------
        # LOAD SCAN
        # -------------------------------------------------
        points = pull_single_scan(bag_path, i, debug)

        # -------------------------------------------------
        # PROCESSING PIPELINE
        # -------------------------------------------------
        LP.update_cloud(points)
        LP.process(False, False)

        # -------------------------------------------------
        # OPEN3D LIVE UPDATE
        # -------------------------------------------------
        ctr = vis.get_view_control()
        ctr.set_zoom(2)
        ctr.set_front([0.4, -0.4, -0.6])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, -1])

        floor_cloud.points = o3d.utility.Vector3dVector(LP.ground_plane_cloud.points)
        ceil_cloud.points = o3d.utility.Vector3dVector(LP.ceiling_cloud.points)
        behind_cloud.points = o3d.utility.Vector3dVector(LP.behind_cloud.points)
        other_cloud.points = o3d.utility.Vector3dVector(LP.obstacle_cloud.points)

        floor_cloud.paint_uniform_color([0, 1, 0])
        ceil_cloud.paint_uniform_color([1, 0, 0])
        behind_cloud.paint_uniform_color([0, 0, 1])
        other_cloud.paint_uniform_color([0.6, 0.6, 0.6])

        vis.update_geometry(floor_cloud)
        vis.update_geometry(ceil_cloud)
        vis.update_geometry(behind_cloud)
        vis.update_geometry(other_cloud)

        vis.poll_events()
        vis.update_renderer()

        if (i % 5 == 0) & debug:
            print("Frame:", i)

    vis.destroy_window()

def run_live_snake_view(bag_path, config : snake_planner.SnakeConfig = None, rail_type="radial", duration=150, start_frame=0, debug=False):
    # =====================================================
    # GRID + SNAKE SETTINGS
    # =====================================================
    L = lidar_processor.LidarProcessor(None)
    GRID_WIDTH_M = L.G_X_MAX - L.G_X_MIN
    GRID_HEIGHT_M = L.G_Y_MAX - L.G_Y_MIN
    del L

    CELL_SIZE = snake_planner.CELL_SIZE

    config = config or snake_planner.SnakeConfig()
    planner = snake_planner.SnakePlanner(config)
    if rail_type == "radial":
        extractor = snake_planner.RadialRailExtractor(config)
    else:
        extractor = snake_planner.StraightRailExtractor(config)

    # =====================================================
    # MATPLOTLIB INTERACTIVE SETUP
    # =====================================================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    img = ax.imshow(
        np.zeros((int(GRID_HEIGHT_M / CELL_SIZE), int(GRID_WIDTH_M / CELL_SIZE))),
        origin='lower', cmap='gray', vmin=0, vmax=1,
        extent=[-GRID_WIDTH_M / 2, GRID_WIDTH_M / 2, 0, GRID_HEIGHT_M]
    )

    ax.arrow(0, 0, 0.0, 0.4,
             width=GRID_WIDTH_M * 0.01,
             head_width=GRID_WIDTH_M * 0.03,
             head_length=GRID_HEIGHT_M * 0.03,
             length_includes_head=True, color='red')

    # one line per rail (free-space band) + one line for the snake
    band_lines = [ax.plot([], [], color="cyan", linewidth=0.5, alpha=0.6)[0]
                  for _ in range(config.num_rails)]
    snake_line, = ax.plot([], [], color="lime", linewidth=2,
                          marker="o", markersize=3)

    ax.set_title(f"Live Snake ({rail_type} rails)")
    ax.set_xlabel("Left / Right (m)")
    ax.set_ylabel("Forward Distance (m)")
    ax.set_aspect('equal')
    plt.show(block=False)

    # =====================================================
    # PLAYBACK CONTROL
    # =====================================================
    paused = False

    def on_key(event):
        nonlocal paused
        if event.key == 'space':
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # =====================================================
    # PROCESS SCANS
    # =====================================================
    num_scans = len(ScanStore(bag_path))
    LP = lidar_processor.LidarProcessor(None)
    start_time = time.time()

    for i in range(start_frame, num_scans):

        if time.time() - start_time > duration:
            print(f"Reached duration limit: {time.time() - start_time:.2f}s")
            break

        while paused:
            plt.pause(0.05)

        if not plt.fignum_exists(fig.number):   # window closed by the user
            break

        # -------------------------------------------------
        # LOAD SCAN + OCCUPANCY GRID
        # -------------------------------------------------
        points = pull_single_scan(bag_path, i, debug)
        LP.update_cloud(points)
        LP.process(False, False)
        grid = LP.occupancy_grid          # <-- adjust to your attribute name

        if i == start_frame and abs(grid.shape[0] * CELL_SIZE - GRID_HEIGHT_M) > 1e-6:
            print(f"WARNING: grid {grid.shape} x CELL_SIZE {CELL_SIZE} != "
                  f"{GRID_HEIGHT_M} m; check CELL_SIZE / extents")

        # -------------------------------------------------
        # SNAKE PIPELINE
        # -------------------------------------------------
        t0 = time.perf_counter()
        rails = extractor.extract_rails(grid)
        planner.plan(rails)
        plan_ms = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------
        # UPDATE PLOT
        # -------------------------------------------------
        img.set_data(grid)

        for line, r in zip(band_lines, rails):
            if r.valid:
                xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
                line.set_data(xs, ys)
            else:
                line.set_data([], [])

        pts = [r.to_xy(v) for r, v in zip(rails, planner.vertices) if r.valid]
        snake_line.set_data([float(p[0]) for p in pts],
                            [float(p[1]) for p in pts])

        ax.set_title(f"Live Snake ({rail_type}) - frame {i} - plan {plan_ms:.1f} ms")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if debug and i % 5 == 0:
            print(f"Frame {i}: {sum(r.valid for r in rails)}/{len(rails)} valid rails, "
                  f"plan {plan_ms:.1f} ms")

        time.sleep(0.01)

    plt.ioff()
    plt.show()

def run_snake_command_view(bag_path, start_frame=0, frame_delay=0.3, duration=1000, debug=False):
    pipeline = NavPipeline()
    num_scans = len(ScanStore(bag_path))
    state = {"paused": False, "step": False}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right" and state["paused"]:
            state["step"] = True

    view, frame, t_start = None, start_frame, time.time()
    print("space = pause/resume, right arrow = step one frame while paused")
    while frame < num_scans and time.time() - t_start < duration:
        if view is not None and not plt.fignum_exists(view.fig.number):
            break
        if state["paused"] and not state["step"]:
            plt.pause(0.05)
            continue
        state["step"] = False

        res = pipeline.process(pull_single_scan(bag_path, frame, debug))
        res["frame"] = frame
        if view is None:
            view = CommandView(res["num_rails"], res["reach_m"], res["grid"].shape)
            view.fig.canvas.mpl_connect("key_press_event", on_key)
        view.update(res)
        frame += 1
        plt.pause(max(frame_delay, 0.001))

    plt.ioff()
    plt.show()

if __name__ == "__main__":

    # points = pull_single_scan(bag_list.extracted_bags[6], 600, True)
    # LP = lidar_processor.LidarProcessor(points, 1000)
    # LP.process(True)
    # EX = snake_planner.RadialRailExtractor(config=snake_planner.SnakeConfig(w_external=3, w_internal=90, num_rails=18,
    #                                             min_dist_m=1, max_dist_m=3.5), debug=True)
    # rails = EX.extract_rails(LP.occupancy_grid)
    # SP = snake_planner.SnakePlanner(debug=True)
    # SP.plan(rails)

    # PP = path_follower.PathFollower()
    # command = PP.compute(rails, SP.vertices, LP.occupancy_grid)

    # local_vis_wrapper.show_height_grid(LP)
    # local_vis_wrapper.plot_snake(SP, LP.occupancy_grid, rails)
    # local_vis_wrapper.plot_blindspot_mask(LP)
    # local_vis_wrapper.plot_guidance_point(SP, PP, LP.occupancy_grid, rails, True)
    # local_vis_wrapper.plot_command_signal(PP)

    run_snake_command_view(bag_list.extracted_bags[8], 500)
