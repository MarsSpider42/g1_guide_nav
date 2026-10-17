"""Laptop side: read the debug stream through the SSH tunnel and show CommandView. No ROS needed here."""
import sys
import matplotlib.pyplot as plt

from debug_client import DebugClient
from nav_state import unpack_state
from command_view import CommandView

STALE_AFTER_S = 1.0


def main(port=5577):
    client = DebugClient("127.0.0.1", port)
    view, last_count, was_stale = None, 0, False
    print(f"waiting for the robot on 127.0.0.1:{port} (is the SSH tunnel up and debug_relay.py running?)")
    while True:
        count, text = client.newest()
        if text is not None and count != last_count:
            last_count = count
            state = unpack_state(text)
            if view is None:
                view = CommandView(state["num_rails"], state["reach_m"], tuple(state["grid_shape"]))
            view.update(state)

        stale = client.seconds_since_rx() > STALE_AFTER_S
        if view is not None and stale != was_stale and hasattr(view, "set_stale"):
            view.set_stale(stale)
            was_stale = stale

        if view is not None and not plt.fignum_exists(view.fig.number):
            break
        plt.pause(0.05)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5577)
