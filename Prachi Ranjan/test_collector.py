"""Run with: python -m unittest test_collector -v (uses no webcam)."""
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from collect_dataset import PEOPLE, SESSIONS, UNKNOWN_ENV, UNKNOWN_STEPS
from collector_gui import CollectorApp, capture_box, crop_image, open_camera, scroll_units
from collector_store import CaptureStore, classmate_indices, images


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.collector-test-', dir=Path(__file__).resolve().parent)
        self.folder = Path(self.temp.name) / 'person' / 's1'
        self.store = CaptureStore(self.folder, 4)
        self.crop = np.full((250, 250, 3), 100, dtype=np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def test_final_pose_and_extras(self):
        for pose in range(4):
            self.store.begin_action()
            for _ in range(5):
                self.store.save(self.crop, pose)
        self.assertEqual(self.store.next_step, 4)
        with self.assertRaises(ValueError):
            self.store.save(self.crop, 3)
        self.store.begin_action()
        self.store.save(self.crop)
        self.assertEqual(self.store.extra_count, 1)
        self.assertEqual(self.store.counts, [5, 5, 5, 5])
        self.assertEqual(cv2.imread(str(images(self.folder)[0])).shape, (250, 250, 3))

    def test_interrupted_burst_resumes_exact_pose(self):
        self.store.save(self.crop, 2)
        self.store.save(self.crop, 2)
        resumed = CaptureStore(self.folder, 4)
        self.assertEqual(resumed.counts, [0, 0, 2, 0])
        self.assertEqual(resumed.next_step, 0)

    def test_failed_image_save_is_not_counted(self):
        with patch('collector_store.cv2.imwrite', return_value=False):
            with self.assertRaises(OSError):
                self.store.save(self.crop, 0)
        self.assertEqual(self.store.counts, [0, 0, 0, 0])
        self.assertEqual(images(self.folder), [])

    def test_failed_progress_write_rolls_back_image(self):
        with patch.object(self.store, '_persist', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.store.save(self.crop, 0)
        self.assertEqual(images(self.folder), [])
        self.assertEqual(self.store.counts[0], 0)

    def test_legacy_and_unique_names(self):
        for i in range(1, 8):
            cv2.imwrite(str(self.folder / f'sample_{i:03d}.jpg'), self.crop)
        store = CaptureStore(self.folder, 4)
        self.assertTrue(store.legacy)
        self.assertEqual(store.counts, [5, 2, 0, 0])
        self.assertEqual(store.save(self.crop, 1).name, 'sample_008.jpg')

    def test_undo_is_session_scoped(self):
        self.store.save(self.crop, 0)
        other = CaptureStore(self.folder.parent / 's2', 4)
        other.save(self.crop, 0)
        self.assertEqual(other.undo(), 1)
        self.assertEqual(len(images(self.folder)), 1)
        self.assertEqual(self.store.undo(), 1)
        self.assertEqual(CaptureStore(self.folder, 4).counts, [0, 0, 0, 0])

    def test_classmate_index_ignores_files(self):
        root = Path(self.temp.name)
        (root / 'classmate_02').mkdir()
        (root / 'classmate_04').mkdir()
        (root / 'classmate_99').touch()
        self.assertEqual(classmate_indices(root), [2, 4])

    def test_crop_at_edge_and_ambiguous_face(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertIsNone(capture_box(frame, []))
        self.assertIsNone(capture_box(frame, [(0, 0, 80, 80)] * 2))
        box = capture_box(frame, [(600, 450, 35, 30)])
        self.assertEqual(crop_image(frame, box, (250, 250)).shape, (250, 250, 3))
        self.assertEqual(capture_box(frame, [], True), (170, 90, 300, 300))

    def test_camera_fallback_releases_failed_backend(self):
        first, second = FakeCamera(), FakeCamera()
        first.read = lambda: (False, None)
        with patch('collector_gui.os.name', 'nt'), patch('collector_gui.cv2.VideoCapture', side_effect=[first, second]):
            cap, frame = open_camera(0)
        self.assertTrue(first.released)
        self.assertIs(cap, second)

    def test_mac_prefers_avfoundation_without_windows_pixel_format(self):
        camera = FakeCamera()
        with patch('collector_gui.sys.platform', 'darwin'), \
                patch('collector_gui.cv2.VideoCapture', return_value=camera) as factory, \
                patch.object(camera, 'set', wraps=camera.set) as properties:
            result, _ = open_camera(2)
        self.assertIs(result, camera)
        factory.assert_called_once_with(2, cv2.CAP_AVFOUNDATION)
        requested = [call.args[0] for call in properties.call_args_list]
        self.assertNotIn(cv2.CAP_PROP_FOURCC, requested)
        self.assertNotIn(cv2.CAP_PROP_BUFFERSIZE, requested)

    def test_mac_warmup_and_fallback(self):
        failed, working = FakeCamera(), FakeCamera()
        failed.read = lambda: (False, None)
        with patch('collector_gui.sys.platform', 'darwin'), \
                patch('collector_gui.time.sleep'), \
                patch('collector_gui.cv2.VideoCapture', side_effect=[failed, working]) as factory:
            camera, _ = open_camera(0)
        self.assertIs(camera, working)
        self.assertTrue(failed.released)
        self.assertEqual([c.args[1] for c in factory.call_args_list], [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY])

    def test_mac_camera_failure_has_permission_guidance(self):
        camera = FakeCamera()
        camera.read = lambda: (False, None)
        with patch('collector_gui.sys.platform', 'darwin'), \
                patch('collector_gui.time.sleep'), \
                patch('collector_gui.cv2.VideoCapture', return_value=camera):
            with self.assertRaisesRegex(RuntimeError, 'Privacy & Security > Camera'):
                open_camera(0)
        self.assertTrue(camera.released)

    def test_mac_small_trackpad_delta_is_not_lost(self):
        self.assertEqual(scroll_units(1, mac=True), -1)
        self.assertEqual(scroll_units(-2, mac=True), 2)
        self.assertEqual(scroll_units(120), -1)
        self.assertEqual(scroll_units(0, mac=True), 0)

    def test_mac_launcher_passes_manual_camera_and_portable_path(self):
        from collect_dataset_mac import main
        location = Path(self.temp.name) / 'new dataset'
        with patch('collector_gui.run') as run, patch('builtins.print'):
            self.assertEqual(main(['--manual', '--camera', '2', '--data-dir', str(location)]), 0)
        self.assertEqual(run.call_args.args[0], location.resolve())
        self.assertTrue(run.call_args.kwargs['manual'])
        self.assertEqual(run.call_args.kwargs['camera_index'], 2)


class FakeCamera:
    def __init__(self):
        self.released = False
        self.frame = np.full((720, 1280, 3), (55, 65, 80), np.uint8)

    def read(self):
        return not self.released, self.frame.copy()

    def release(self):
        self.released = True

    def set(self, *_):
        return True

    def isOpened(self):
        return not self.released


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.collector-test-', dir=Path(__file__).resolve().parent)
        self.root = tk.Tk()
        self.root.withdraw()
        self.camera = FakeCamera()
        with patch('collector_gui.open_camera', return_value=(self.camera, self.camera.frame)):
            self.app = CollectorApp(self.root, Path(self.temp.name), PEOPLE, SESSIONS,
                                    UNKNOWN_STEPS, UNKNOWN_ENV, (250, 250), 5, 0.2, 3, True)
        self.app._tick()

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def save_action(self, n=1, extra=False):
        self.app.last_save = -float('inf')
        self.app.start_capture(n, extra=extra)
        for _ in range(n):
            self.app.next_shot = 0
            self.app._tick()

    def test_single_manual_and_extra_burst(self):
        self.save_action()
        self.save_action(5, extra=True)
        self.assertEqual(self.app.store.extra_count, 6)
        self.assertEqual(self.app.store.counts, [0] * 4)

    def test_guided_completion_and_revisit_pose(self):
        self.app.plan.set('Guided poses')
        self.app.pose_combo.current(2)
        self.app.change_pose()
        self.save_action(5)
        self.assertEqual(self.app.store.counts, [0, 0, 5, 0])
        for _ in range(3):
            self.save_action(5)
        self.assertEqual(self.app.store.next_step, 4)
        self.app._tick()  # regression: final burst must not index steps[4]
        self.save_action(extra=True)
        self.assertEqual(self.app.store.extra_count, 1)

    def test_unknown_next_previous_and_new(self):
        self.app.member.set('5: Unknown')
        self.app.change_person()
        self.assertEqual(self.app.classmate, 1)
        self.app.change_session(1)
        self.assertEqual(self.app.classmate, 2)
        self.app.change_session(-1)
        self.assertEqual(self.app.classmate, 1)
        self.app.new_classmate()
        self.assertEqual(self.app.classmate, 3)
        self.assertEqual(self.app.store.steps, 2)

    def test_stop_partial_burst_and_resume(self):
        self.app.plan.set('Guided poses')
        self.app.start_capture(5)
        self.app._tick()
        self.app.stop()
        self.assertEqual(self.app.store.counts[0], 1)
        self.save_action(5)
        self.assertEqual(self.app.store.counts[0], 5)
        self.assertEqual(self.app.pose, 1)

    def test_lost_face_does_not_save_old_box(self):
        self.app.crop_mode.set('Detected face')
        self.app.detectors = (None, None)
        with patch('collector_gui.detect_faces', return_value=[]):
            self.app._tick()
        self.app.start_capture(1)
        self.assertEqual(self.app.remaining, 0)
        self.assertIsNone(self.app.box)

    def test_mac_command_shortcuts_do_not_capture(self):
        with patch('collector_gui.sys.platform', 'darwin'):
            self.app.on_key(SimpleNamespace(state=0x08, widget=self.app.video, keysym='c'))
        self.assertEqual(self.app.remaining, 0)

    def test_bad_session_metadata_restores_context(self):
        folder = self.app.base / 'Rahul' / 's2'
        folder.mkdir(parents=True)
        (folder / '_collector_progress.json').write_text('broken')
        self.app.change_session(1)
        self.assertEqual(self.app.session, 0)
        self.assertEqual(self.app.store.folder, self.app.base / 'Rahul' / 's1')
        self.assertIn('Cannot open', self.app.status.get())


if __name__ == '__main__':
    unittest.main()
