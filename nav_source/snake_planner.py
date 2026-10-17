from dataclasses import dataclass
from typing import Any
import numpy as np
#import matplotlib.pyplot as plt

CELL_SIZE = 0.05 #m/cell, dependent on the constant in LidarProcessor

@dataclass
class Rail:
    left_bound: float
    right_bound: float
    valid: bool
    left_closed : bool
    right_closed : bool

    def __init__(self, left_bound, right_bound, valid, left_closed=True, right_closed=True) -> None:
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.valid = valid
        self.left_closed = left_closed
        self.right_closed = right_closed

    @property
    def coordinate(self) -> float:
        return 0.0

    @property
    def lower(self) -> float:
        return -self.left_bound

    @property
    def upper(self) -> float:
        return self.right_bound

@dataclass
class Straight_Rail(Rail): # datatype containing all of the details of an individual straight rail.
    distance_m: float
    row: int

    def __init__(self, distance_m, row, left_bound, right_bound, valid, left_closed=True, right_closed=True) -> None:
        super().__init__(left_bound, right_bound, valid, left_closed, right_closed)
        self.distance_m = distance_m
        self.row = row

    @property
    def coordinate(self) -> float:
        return self.distance_m

    def to_xy(self, vertex):
        v = np.asarray(vertex, dtype=float)
        return v, np.full_like(v, self.distance_m)

@dataclass
class Radial_Rail(Rail): # datatype containing all of the details of an individual radial rail
    radius_m: float

    def __init__(self, radius_m, left_bound, right_bound, valid, left_closed=True, right_closed=True) -> None:
        super().__init__(left_bound, right_bound, valid, left_closed, right_closed)
        self.radius_m = radius_m

    @property
    def coordinate(self) -> float:
        return self.radius_m

    def to_xy(self, vertex):
        v = np.asarray(vertex, dtype=float)
        return self.radius_m * np.sin(v), self.radius_m * np.cos(v)

@dataclass
class SnakeConfig: # datatype containing the configuration options of the snake planner
    num_rails: int = 12
    min_dist_m: float = 0.3
    max_dist_m: float = 6.0
    w_external: float = 3.0
    w_internal: float = 1.0
    w_potential: float = 0.0
    bias_direction: float = 0.0
    open_min_half_width_m: float = 0.5

class StraightRailExtractor:
    def __init__(self, config: SnakeConfig = None):
        self.config = config or SnakeConfig()

    def extract_rails(self, grid: np.ndarray) -> list:
        origin_row, origin_col = SnakePlanner.robot_origin(grid)
        distances = np.linspace(
            self.config.min_dist_m, self.config.max_dist_m, self.config.num_rails
        )
        rails = []

        for d in distances:
            row = origin_row + int(round(d / CELL_SIZE))
            if row >= grid.shape[0]:
                rails.append(Straight_Rail(d, row, 0.0, 0.0, False))
                continue

            row_data = grid[row, :]
            free_cols = np.where(row_data == 0)[0]

            if free_cols.size == 0:
                rails.append(Straight_Rail(d, row, 0.0, 0.0, False))
                continue

            run = self._contiguous_run_containing(free_cols, origin_col)
            if run is None:
                rails.append(Straight_Rail(d, row, 0.0, 0.0, False))
                continue

            left_col, right_col = run
            left_m = (origin_col - left_col) * CELL_SIZE
            right_m = (right_col - origin_col) * CELL_SIZE
            rails.append(Straight_Rail(d, row, left_m, right_m, True))

        return rails

    @staticmethod
    def _contiguous_run_containing(free_cols: np.ndarray, target_col: int):
        if target_col not in free_cols:
            target_col = free_cols[np.argmin(np.abs(free_cols - target_col))]
        idx = np.searchsorted(free_cols, target_col)
        left = idx
        while left > 0 and free_cols[left - 1] == free_cols[left] - 1:
            left -= 1
        right = idx
        while right < len(free_cols) - 1 and free_cols[right + 1] == free_cols[right] + 1:
            right += 1
        return free_cols[left], free_cols[right]


class RadialRailExtractor:

    def __init__(self, config: SnakeConfig = None, angle_range_deg: float = 90.0, angle_step_deg: float = 1.0, debug=False):
        self.config = config or SnakeConfig()
        self.angle_range_rad = np.radians(angle_range_deg)
        self.angle_step_rad = np.radians(angle_step_deg)
        self.debug = debug
        self.FREE = 0
        self.OCCUPIED = 1
        self.OUTSIDE = 2
        self._key = None
        self._cache = None

    def extract_rails(self, grid: np.ndarray) -> list:
        origin_row, origin_col = SnakePlanner.robot_origin(grid)

        key = (grid.shape, self.config.num_rails, self.config.min_dist_m, self.config.max_dist_m,
               self.angle_range_rad, self.angle_step_rad)
        if key != self._key:                      # rebuild only if the config or grid shape changed
            radii = np.linspace(self.config.min_dist_m, self.config.max_dist_m, self.config.num_rails)
            angles = np.arange(-self.angle_range_rad, self.angle_range_rad + 1e-9, self.angle_step_rad)
            R, A = np.meshgrid(radii, angles, indexing="ij")
            rows = origin_row + np.rint(R * np.cos(A) / CELL_SIZE).astype(int)
            cols = origin_col + np.rint(R * np.sin(A) / CELL_SIZE).astype(int)
            inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
            self._cache = (radii, angles, np.where(inside, rows, 0), np.where(inside, cols, 0), inside)
            self._key = key
        radii, angles, rows, cols, inside = self._cache

        states = np.where(inside, np.where(grid[rows, cols] != 0, self.OCCUPIED, self.FREE),
                          self.OUTSIDE).astype(np.int8)

        rails = []
        for k, radius in enumerate(radii):
            state = states[k]
            free_flags = (state == self.FREE)

            run = self._contiguous_run_containing(angles, free_flags)
            if run is None:
                rails.append(Radial_Rail(radius, 0.0, 0.0, False))
                continue

            left, right = run
            angle_min, angle_max = angles[left], angles[right]
            left_closed = bool(left > 0 and state[left - 1] == self.OCCUPIED)
            right_closed = bool(right < len(angles) - 1 and state[right + 1] == self.OCCUPIED)

            # (your commented-out virtual-wall block can stay here)

            rails.append(Radial_Rail(radius, -angle_min, angle_max, True,
                                     left_closed, right_closed))

        no_space = [not r.valid for r in rails]

        count_closed = 0
        for r in rails:
            if ((r.left_closed == True) & (r.right_closed == True)):
                count_closed += 1

        for r in rails:
            if(((r.left_closed == False) | (r.right_closed == False)) & ((count_closed / self.config.num_rails) > 0.40)):
                r.valid = False

        if self.debug:
            frac = count_closed / self.config.num_rails
            mode = "CORRIDOR" if frac > 0.40 else "OPEN AREA"
            print(f"\n[rails] {count_closed}/{self.config.num_rails} closed ({frac:.0%}) -> {mode} mode")
            print(f"{'idx':>3} {'dist (m)':>8}  {'left':>6}  {'right':>6}  {'rail':>6}  {'band (deg)':>16}  status")
            for i, r in enumerate(rails):
                left  = "closed" if r.left_closed else "open"
                right = "closed" if r.right_closed else "open"
                state = "closed" if (r.left_closed and r.right_closed) else "open"
                if no_space[i]:
                    band, status = "-", "INVALID (no free space)"
                else:
                    # negative = left of forward, positive = right
                    band = f"{-np.degrees(r.left_bound):6.1f} to {np.degrees(r.right_bound):6.1f}"
                    status = "valid" if r.valid else "INVALID (dropped: corridor rule)"
                print(f"{i:>3} {r.radius_m:>8.2f}  {left:>6}  {right:>6}  {state:>6}  {band:>16}  {status}")

        return rails

    @staticmethod
    def _contiguous_run_containing(angles: np.ndarray, free_flags: np.ndarray):
        zero_idx = int(np.argmin(np.abs(angles)))  # index nearest angle=0 (straight ahead)

        if not free_flags[zero_idx]:
            free_idx = np.where(free_flags)[0]
            if free_idx.size == 0:
                return None
            zero_idx = free_idx[np.argmin(np.abs(free_idx - zero_idx))]

        left = zero_idx
        while left > 0 and free_flags[left - 1]:
            left -= 1
        right = zero_idx
        while right < len(free_flags) - 1 and free_flags[right + 1]:
            right += 1

        return left, right



class SnakePlanner:
    def __init__(self, config: SnakeConfig = None, debug : bool = False):
        self.config = config or SnakeConfig()
        self.debug = debug

    def plan(self, rails : list):
        """ Entry to full pipeline"""
        self.initialise_snake(rails)
        self.relax_snake(rails)

    @staticmethod
    def robot_origin(grid: np.ndarray) -> tuple:
        """Robot at row 0 (forward=dimension 0), in centre column"""
        return 0, grid.shape[1] // 2 

    @staticmethod
    def external_energy(offset: float, rail : Straight_Rail) -> float:
        '''Repulsive energy: rises sharply as the vertex nears either boundary of the rail.'''

        if not rail.valid:
            return 1e6

        margin_left = offset - (-rail.left_bound) # distance from vertex to left end
        margin_right = rail.right_bound - offset # distance from vertex to right end
        margin = min(margin_left, margin_right) # find the closest wall

        if margin <= 0:
            return 1e6

        return 1.0/(margin**2) # weighted to grow as the closeness of either wall increases

    @staticmethod
    def internal_energy(offsets : np.ndarray, rails : list) -> float:
        '''Penalise curvature: Calculate sum of squared second differences.'''
        xs = offsets
        ys = np.array([r.coordinate for r in rails])

        if len(xs) < 3:
            return 0.0

        # discrete second derivative in x and y
        d2x = xs[2:] - 2*xs[1:-1] + xs[:-2] 
        d2y = ys[2:] - 2*ys[1:-1] + ys[:-2]
        return float(np.sum(d2x**2 + d2y**2)) # sum of squares

    def potential_energy(self, offsets: np.ndarray) -> float:
        '''Optional driving hint: bias the whole snake towards one side'''
        return float(-self.config.w_potential * np.sum(self.config.bias_direction * offsets))

    def total_energy(self, offsets: np.ndarray, rails: list) -> float:
        ext = sum(self.external_energy(o,r) for o,r in zip(offsets, rails))
        intl = self.internal_energy(offsets, rails)
        pot = self.potential_energy(offsets)
        return self.config.w_external * ext + self.config.w_internal * intl + pot

    def initialise_snake(self, rails : list):
        '''Start each vertex at its rail's free-space midpoint.'''
        vertices = np.zeros(len(rails))

        for i, rail in enumerate(rails):
            if rail.valid:
                vertices[i] = (rail.right_bound - rail.left_bound) / 2.0

        self.vertices = vertices


    def _local_cost(self, i, candidates, rails):
        """Cost of vertex i at each candidate offset, other vertices fixed."""
        c, r, v = self.config, rails[i], self.vertices

        # wall repulsion: vectorised copy of external_energy
        margin = np.minimum(candidates - r.lower, r.upper - candidates)
        ext = np.where(margin > 0, 1.0 / np.maximum(margin, 1e-12) ** 2, 1e6)

        def x(k):                            # vertex k, with vertex i replaced by the candidates
            return candidates if k == i else v[k]

        intl = 0.0
        for j in (i - 1, i, i + 1):          # only these second differences involve vertex i
            if 1 <= j <= len(v) - 2:
                intl = intl + (x(j + 1) - 2 * x(j) + x(j - 1)) ** 2
        # the d2y terms don't depend on the offsets, so they are constant and dropped

        pot = -c.w_potential * c.bias_direction * candidates
        return c.w_external * ext + c.w_internal * intl + pot

    def relax_snake(self, rails: list, iterations: int = 30, search_steps: int = 21):
        self.passes = 0                      # how many passes this frame needed (for diagnostics)
        for _ in range(iterations):
            self.passes += 1
            moved = False
            for i, r in enumerate(rails):
                if not r.valid:
                    continue
                candidates = np.linspace(r.lower, r.upper, search_steps)
                best = candidates[np.argmin(self._local_cost(i, candidates, rails))]
                if not np.isclose(best, self.vertices[i]):
                    moved = True
                self.vertices[i] = best
            if not moved:
                break
