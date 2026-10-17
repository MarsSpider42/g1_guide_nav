"""Background reader for the newline-delimited debug stream. Keeps only the newest line, reconnects forever."""
import socket
import threading
import time


class DebugClient:
    def __init__(self, host="127.0.0.1", port=5577):
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.latest, self.count, self.last_rx = None, 0, 0.0
        self.connected = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=3) as s:
                    s.settimeout(None)
                    self.connected = True
                    for line in s.makefile("rb"):
                        with self.lock:
                            self.latest, self.count, self.last_rx = line.decode(), self.count + 1, time.monotonic()
            except OSError:
                pass
            self.connected = False
            time.sleep(1.0)

    def newest(self):
        """(count, text) of the newest line; count increases with every line received."""
        with self.lock:
            return self.count, self.latest

    def seconds_since_rx(self):
        return time.monotonic() - self.last_rx if self.last_rx else float("inf")
