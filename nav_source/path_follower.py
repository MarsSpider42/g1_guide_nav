import numpy as np
from dataclasses import dataclass


@dataclass
class DriveConfig:
    guidance_rail: int = 6          # 0-based, nearest first -> the 5th rail
    # minimum non-zero commands: below these the motors don't respond
    min_vx: float = 0.4             # m/s
    min_omega: float = 0.7          # rad/s
    # upper limits: unknown, set these once you know them (equal to min = fixed speed)
    max_vx: float = 0.4
    max_omega: float = 0.7
    k_yaw: float = 1.0              # rad/s per rad of error, only matters if max_omega > min_omega
    turn_on_angle: float = np.radians(8)        # start turning above this heading error
    turn_off_angle: float = np.radians(4)        # stop turning below this (hysteresis)
    turn_in_place_angle: float = np.radians(30)  # above this, stop walking and just turn
    smoothing: float = 0.5          # weight of the new heading error: 1.0 = no smoothing


@dataclass
class DriveCommand:
    vx: float = 0.0       # m/s, + forward
    vy: float = 0.0       # m/s, + left
    omega: float = 0.0    # rad/s, + counter-clockwise (turn left)
    status: str = "ok"


class PathFollower:
    def __init__(self, config: DriveConfig = None):
        self.config = config or DriveConfig()
        self.err_f = None         # filtered heading error
        self.turning = False      # hysteresis state
        self.target_xy = None     # (right, forward) in metres, kept for plotting
        self.last_cmd = None

    def _stop(self, status: str) -> DriveCommand:
        self.err_f = None
        self.turning = False
        self.last_cmd = DriveCommand(status=status)
        return self.last_cmd

    
    def _path_is_clear(self,occupancy_grid: np.ndarray,x_right: float,y_fwd: float,clearance_m: float = 0.10,sample_step_m: float = 0.05) -> bool:
        """Check a straight-line path against the occupancy grid."""

        grid = np.asarray(occupancy_grid)
        cell_size = 0.05

        if grid.ndim != 2 or grid.size == 0:
            return False

        if y_fwd <= 0:
            return False

        origin_col = grid.shape[1] // 2
        radius_cells = int(np.ceil(clearance_m / cell_size))

        distance = np.hypot(x_right, y_fwd)
        n_samples = max(1, int(np.ceil(distance / sample_step_m)))

        for t in np.linspace(0.0, 1.0, n_samples + 1):
            x = t * x_right
            y = t * y_fwd

            # Skip the robot's immediate origin, where its own footprint
            # or LiDAR blind zone may interfere with the occupancy grid.
            if y < clearance_m:
                continue

            row = int(round(y / cell_size))
            col = int(round(origin_col + x / cell_size))

            # A path outside the lateral grid limits is unsafe.
            if col - radius_cells < 0 or col + radius_cells >= grid.shape[1]:
                return False

            # Clip the forward clearance window to the grid.
            # Do not reject a path merely because the window extends
            # behind row 0 or beyond the last row.
            r0 = max(0, row - radius_cells)
            r1 = min(grid.shape[0], row + radius_cells + 1)
            c0 = col - radius_cells
            c1 = col + radius_cells + 1

            if r0 >= r1:
                return False

            if np.any(grid[r0:r1, c0:c1] != 0):
                return False

        return True


    def compute(self, rails: list, vertices: np.ndarray, occupancy_grid: np.ndarray) -> DriveCommand:
        c = self.config
        self.target_xy = None

        c = self.config
        self.target_xy = None
        original = c.guidance_rail

        # 1. Keep the existing guidance-rail validity check
        preferred = c.guidance_rail

        if (
            preferred < 0
            or preferred >= len(rails)
            or not rails[preferred].valid
        ):
            next_valid = None

            # Search from the preferred rail onwards first.
            search_order = list(range(max(0, preferred), len(rails)))

            # Then consider earlier rails.
            search_order += list(range(min(preferred, len(rails)), 0))
            search_order += [0] if rails else []

            for idx in search_order:
                if rails[idx].valid:
                    next_valid = idx
                    break

            if next_valid is None:
                c.guidance_rail = original
                return self._stop("no guidance rail")

            c.guidance_rail = next_valid

        # 2. Check the selected guidance point for obstacles
        preferred = c.guidance_rail

        candidate_indices = sorted(
            range(len(rails)),
            key=lambda i: (abs(i - preferred), i)
        )

        selected = None

        for i in candidate_indices:
            if not rails[i].valid or i >= len(vertices):
                continue

            x_right, y_fwd = (
                float(a) for a in rails[i].to_xy(vertices[i])
            )

            if self._path_is_clear(occupancy_grid, x_right, y_fwd):
                selected = (i, x_right, y_fwd)
                break

        if selected is None:
            c.guidance_rail = original
            return self._stop("no collision-free guidance point")

        rail_index, x_right, y_fwd = selected
        self.target_xy = (x_right, y_fwd)

        # 2. grid frame (x right, y forward) -> robot frame (forward, left)
        forward, left = y_fwd, -x_right
        if np.hypot(forward, left) < 1e-6:
            c.guidance_rail = original
            return self._stop("target at robot")

        heading_err = np.arctan2(left, forward)       # + = target is to the left
        a = c.smoothing
        self.err_f = heading_err if self.err_f is None else a * heading_err + (1 - a) * self.err_f
        err = self.err_f

        # 3. omega: on/off with hysteresis; when on, never below min_omega
        if self.turning:
            self.turning = abs(err) > c.turn_off_angle
        else:
            self.turning = abs(err) > c.turn_on_angle

        omega = 0.0
        if self.turning:
            omega = float(np.sign(err) * np.clip(c.k_yaw * abs(err), c.min_omega, c.max_omega))

        # 4. vx: 0 when badly misaligned, otherwise never below min_vx
        if abs(err) > c.turn_in_place_angle:
            vx = 0.0
        else:
            scale = 1.0 - abs(err) / c.turn_in_place_angle   # 1 straight ahead -> 0 at the limit
            vx = float(np.clip(c.max_vx * scale, c.min_vx, c.max_vx))

        c.guidance_rail = original

        self.last_cmd = DriveCommand(vx, 0.0, omega)
        return self.last_cmd