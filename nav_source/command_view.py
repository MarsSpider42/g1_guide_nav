import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from snake_planner import CELL_SIZE          # metres per cell, must match LidarProcessor
from robot_marker import add_robot, ROBOT_DEPTH_M

MODE_COLORS = {"drive": "lime", "arc": "orange", "pivot": "red", "stop": "lightgrey"}


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


class CommandView:
    """Two panes: grid + rails + snake + guidance point (left), robot + commands only (right)."""

    def __init__(self, num_rails, reach_m, grid_shape=(120, 120), max_vx=0.4, max_omega=0.7):
        self.max_vx, self.max_omega = max_vx, max_omega
        depth_m, width_m = grid_shape[0] * CELL_SIZE, grid_shape[1] * CELL_SIZE

        plt.ion()
        self.fig, (self.ax_map, self.ax_cmd) = plt.subplots(1, 2, figsize=(15, 8))

        # ---- left pane ----
        ax = self.ax_map
        self.img = ax.imshow(np.zeros(grid_shape), origin="lower", cmap="gray", vmin=0, vmax=1,
                             extent=[-width_m / 2, width_m / 2, 0, depth_m])
        self.band_lines = [ax.plot([], [], color="cyan", linewidth=1.0, alpha=0.8)[0]
                           for _ in range(num_rails)]
        self.snake_line, = ax.plot([], [], color="lime", linewidth=2, marker="o", markersize=3)
        self.guide_path, = ax.plot([], [], color="yellow", linewidth=1.5, linestyle=":")   # straight path checked
        self.guide_dot, = ax.plot([], [], marker="*", linestyle="none", color="yellow",
                                  markersize=18, markeredgecolor="black", zorder=4)
        add_robot(ax)
        ax.set_xlabel("Left / Right (m)")
        ax.set_ylabel("Forward Distance (m)")
        ax.set_aspect("equal")
        ax.legend(handles=[
            Line2D([0], [0], color="cyan", lw=1, label="Closed rail"),
            Line2D([0], [0], color="orange", lw=1, label="Open rail (used)"),
            Line2D([0], [0], color="orange", lw=1, ls="--", label="Open rail (dropped)"),
            Line2D([0], [0], color="lime", lw=2, marker="o", markersize=3, label="Snake"),
            Line2D([0], [0], color="yellow", marker="*", markersize=14, lw=0,
                   markeredgecolor="black", label="Guidance point"),
            Line2D([0], [0], color="yellow", lw=1.5, ls=":", label="Checked path"),
        ], loc="upper right", fontsize=8)

        # ---- right pane ----
        ax = self.ax_cmd
        add_robot(ax)                                  # call first: it adjusts the y limits
        ax.set_facecolor("black")
        ax.set_xlim(-reach_m, reach_m)
        ax.set_ylim(0, reach_m)
        ax.set_aspect("equal")
        ax.grid(True, color="0.25", linewidth=0.5)
        ax.set_xlabel("Left / Right (m)")
        ax.set_ylabel("Forward Distance (m)")
        ax.set_title("Commands the planner would send (display only)")

        self.guide_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=22,
                                           color="yellow", linewidth=2.5, zorder=3)   # robot -> guidance point
        self.vx_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=22,
                                        color="lime", linewidth=5, zorder=4)          # forward speed
        self.turn_head = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=22,
                                         color="orange", linewidth=3, zorder=4)       # turn arc arrowhead
        self.turn_arc, = ax.plot([], [], color="orange", linewidth=3, zorder=4)
        for p in (self.guide_arrow, self.vx_arrow, self.turn_head):
            p.set_visible(False)
            ax.add_patch(p)
        self.cmd_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                                fontsize=14, family="monospace", color="white",
                                bbox=dict(boxstyle="round", facecolor="black", alpha=0.75,
                                          edgecolor="white"))

    def _update_turn_arc(self, omega):
        if omega == 0:
            self.turn_arc.set_data([], [])
            self.turn_head.set_visible(False)
            return
        sweep = np.radians(60) * min(1.0, abs(omega) / self.max_omega)
        phi = np.pi / 2 + np.sign(omega) * np.linspace(0.0, sweep, 30)    # + omega = left = counter-clockwise
        xs, ys = 0.5 * np.cos(phi), 0.5 * np.sin(phi)
        self.turn_arc.set_data(xs, ys)
        set_arrow(self.turn_head, (xs[-3], ys[-3]), (xs[-1], ys[-1]))

    def update(self, s):
        grid, rails, vertices, cmd = s["grid"], s["rails"], s["vertices"], s["cmd"]
        tgt, err = s["target_xy"], s["err"]

        # ---- left pane ----
        self.img.set_data(grid)
        for k, line in enumerate(self.band_lines):
            r = rails[k] if k < len(rails) else None
            is_open = r is not None and not (r.left_closed and r.right_closed)
            # draw valid rails, plus open rails dropped by the corridor rule;
            # skip rails with no free space (they default to "closed" and invalid)
            if r is not None and (r.valid or is_open):
                xs, ys = r.to_xy(np.linspace(r.lower, r.upper, 50))
                line.set_data(xs, ys)
                line.set_color("orange" if is_open else "cyan")
                line.set_linestyle("--" if (is_open and not r.valid) else "-")
            else:
                line.set_data([], [])
        pts = [r.to_xy(v) for r, v in zip(rails, vertices) if r.valid]
        self.snake_line.set_data([float(p[0]) for p in pts], [float(p[1]) for p in pts])
        if tgt:
            self.guide_dot.set_data([tgt[0]], [tgt[1]])
            self.guide_path.set_data([0, tgt[0]], [0, tgt[1]])
        else:
            self.guide_dot.set_data([], [])
            self.guide_path.set_data([], [])
        parts = [f"frame {s.get('frame', '-')}"] + [f"{k[:-3]} {v:.0f} ms" for k, v in s.get("timing", {}).items()]
        self.ax_map.set_title(" | ".join(parts))

        # ---- right pane ----
        m = mode_of(cmd)
        set_arrow(self.guide_arrow, (0.0, 0.0), tgt)
        front = ROBOT_DEPTH_M / 2
        set_arrow(self.vx_arrow, (0.0, front),
                  (0.0, front + cmd.vx / self.max_vx) if cmd.vx > 0 else None)   # 1 m long at max speed
        self._update_turn_arc(cmd.omega)

        turn = "left" if cmd.omega > 0 else "right" if cmd.omega < 0 else "-"
        err_txt = "-" if err is None else f"{np.degrees(err):+.1f} deg"
        lines = [m.upper(), f"vx    {cmd.vx:+.2f} m/s", f"omega {cmd.omega:+.2f} rad/s ({turn})",
                 f"error {err_txt}"]
        if cmd.status != "ok":
            lines.append(cmd.status)
        self.cmd_text.set_text("\n".join(lines))
        self.cmd_text.set_color(MODE_COLORS[m])
        self.fig.canvas.draw_idle()

    def set_stale(self, stale):
        self.fig.suptitle("NO DATA FROM ROBOT" if stale else "", color="red", fontsize=16)
        self.fig.canvas.draw_idle()