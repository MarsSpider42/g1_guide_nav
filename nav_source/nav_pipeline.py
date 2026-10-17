import time
import numpy as np
from lidar_processor import LidarProcessor
from snake_planner import SnakeConfig, SnakePlanner, RadialRailExtractor
from path_follower import PathFollower


class NavPipeline:
    """points (N, 3) -> occupancy grid -> rails -> snake -> drive command. No ROS in here."""

    def __init__(self, config=None, drive_config=None):
        self.config = config or SnakeConfig(w_external=3, w_internal=90, num_rails=18,
                                            min_dist_m=1, max_dist_m=3.5)
        self.lp = LidarProcessor(None)
        self.extractor = RadialRailExtractor(self.config)
        self.planner = SnakePlanner(self.config)
        self.follower = PathFollower(drive_config)
        self.grid_shape = (120, 120)            # updated after the first good frame

    def process(self, points):
        timing = {}
        try:
            t0 = time.perf_counter()
            self.lp.update_cloud(points)
            self.lp.process(False)
            grid = np.array(self.lp.occupancy_grid)
            self.grid_shape = grid.shape
            timing["data_process_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            rails = self.extractor.extract_rails(grid)
            self.planner.plan(rails)
            vertices = self.planner.vertices.copy()
            cmd = self.follower.compute(rails, self.planner.vertices, grid)
            timing["snake_plan_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:                  # any failure (e.g. "No valid ground plane found") = a stop
            grid, rails, vertices = np.zeros(self.grid_shape, np.uint8), [], np.zeros(0)
            cmd = self.follower._stop(f"processing failed: {e}")
            self.follower.target_xy = None

        return {"grid": grid, "rails": rails, "vertices": vertices, "cmd": cmd,
                "target_xy": self.follower.target_xy, "err": self.follower.err_f,
                "timing": timing, "num_rails": self.config.num_rails,
                "reach_m": self.config.max_dist_m + 0.3}