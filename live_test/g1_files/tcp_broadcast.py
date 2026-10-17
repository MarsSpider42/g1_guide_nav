"""Tiny newline-delimited TCP broadcaster. Each client only ever gets the NEWEST message:
a slow client skips frames instead of building up a backlog. No ROS in here."""
import socket
import threading


class _Client(threading.Thread):
    def __init__(self, conn, on_close):
        super().__init__(daemon=True)
        self.conn, self.on_close = conn, on_close
        self.cond = threading.Condition()
        self.latest = None
        self.alive = True

    def push(self, data: bytes):
        with self.cond:
            self.latest = data              # overwrite: older unsent frames are dropped
            self.cond.notify()

    def run(self):
        try:
            while self.alive:
                with self.cond:
                    while self.latest is None and self.alive:
                        self.cond.wait(0.5)
                    data, self.latest = self.latest, None
                if data is not None:
                    self.conn.sendall(data)
        except OSError:
            pass
        finally:
            self.alive = False
            try:
                self.conn.close()
            except OSError:
                pass
            self.on_close(self)


class TcpBroadcaster:
    def __init__(self, port=5577, host="127.0.0.1"):
        self.clients, self.lock = [], threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))          # localhost only: reached through the SSH tunnel
        self.server.listen(4)
        self.port = port
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c = _Client(conn, self._remove)
            with self.lock:
                self.clients.append(c)
            c.start()

    def _remove(self, c):
        with self.lock:
            if c in self.clients:
                self.clients.remove(c)

    def send(self, text: str):
        data = text.encode() + b"\n"
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            c.push(data)

    @property
    def num_clients(self):
        return len(self.clients)
