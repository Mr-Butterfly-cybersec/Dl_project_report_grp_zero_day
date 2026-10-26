"""Image storage and per-pose progress; independent of camera and GUI."""
import json
import time
from pathlib import Path

import cv2

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.exists() else []


def classmate_indices(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(int(p.name[10:]) for p in root.iterdir()
                  if p.is_dir() and p.name.startswith('classmate_') and p.name[10:].isdigit())


class CaptureStore:
    def __init__(self, folder, steps, shots=5):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.steps, self.shots = steps, shots
        self.manifest = self.folder / '_collector_progress.json'
        self.entries = {}
        self.legacy = not self.manifest.exists() and bool(images(self.folder))
        if self.manifest.exists():
            data = json.loads(self.manifest.read_text(encoding='utf-8'))
            entries = data['images']
            if not isinstance(entries, dict) or any(
                Path(name).name != name or not isinstance(step, int) or
                not 0 <= step < steps for name, step in entries.items()
            ):
                raise ValueError(f'Invalid progress data: {self.manifest}')
            self.entries = {name: step for name, step in entries.items()
                            if (self.folder / name).is_file()}
        else:
            # Old collector had no pose metadata: preserve its sequential assumption.
            old = sorted(images(self.folder), key=self._order)
            self.entries = {p.name: i // shots for i, p in enumerate(old[:steps * shots])}
        self.history = []

    @staticmethod
    def _order(path):
        tail = path.stem.rsplit('_', 1)[-1]
        return (int(tail) if tail.isdigit() else float('inf'), path.name)

    @property
    def counts(self):
        return [sum(step == i for step in self.entries.values()) for i in range(self.steps)]

    @property
    def next_step(self):
        return next((i for i, count in enumerate(self.counts) if count < self.shots), self.steps)

    @property
    def extra_count(self):
        return len(images(self.folder / 'extras'))

    def _persist(self):
        temp = self.manifest.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'version': 1, 'images': self.entries}, indent=2), encoding='utf-8')
        # Windows indexing/virus scanners can briefly hold a newly written file.
        for attempt in range(5):
            try:
                temp.replace(self.manifest)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (2 ** attempt))

    def begin_action(self):
        self.history.append([])

    def save(self, crop, step=None):
        if step is not None and (not 0 <= step < self.steps or self.counts[step] >= self.shots):
            raise ValueError('This pose is complete. Choose another pose or use Extra photo.')
        folder = self.folder if step is not None else self.folder / 'extras'
        folder.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.stem[7:]) for p in images(folder)
                   if p.stem.startswith('sample_') and p.stem[7:].isdigit()]
        path = folder / f'sample_{max(numbers, default=0) + 1:03d}.jpg'
        # Exclusive reservation avoids replacing a file if two collectors are open.
        with path.open('xb'):
            pass
        try:
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f'Could not save {path}')
            if step is not None:
                self.entries[path.name] = step
                self._persist()
        except Exception:
            if step is not None:
                self.entries.pop(path.name, None)
            path.unlink(missing_ok=True)
            raise
        if not self.history:
            self.begin_action()
        self.history[-1].append(path)
        return path

    def undo(self):
        while self.history and not self.history[-1]:
            self.history.pop()
        if not self.history:
            return 0
        action = self.history[-1]
        removed = 0
        for path in list(action):
            # A failed deletion stays in history so it can be retried.
            path.unlink(missing_ok=True)
            if path.parent == self.folder:
                self.entries.pop(path.name, None)
            action.remove(path)
            removed += 1
        self._persist()
        self.history.pop()
        return removed
