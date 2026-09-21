"""Process-local, immutable NPZ snapshot shared by conversion and source audits."""

import hashlib
import io
from pathlib import Path

import numpy as np


class BoltzSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.identity = self._identity()
        payload = self.path.read_bytes()
        self.size = len(payload)
        self.sha256 = hashlib.sha256(payload).hexdigest()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        for array in self.data.values():
            array.flags.writeable = False
        self.nbytes = sum(a.nbytes for a in self.data.values())
        self.memo = {}
        self.check_path(path)

    def _identity(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def check_path(self, path):
        if Path(path).resolve() != self.path or self._identity() != self.identity:
            raise ValueError("source NPZ changed after cached parsing")
