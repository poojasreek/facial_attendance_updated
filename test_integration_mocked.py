"""
test_integration_mocked.py
Mocked camera and recognition tests to verify end-to-end workflows
without requiring physical camera hardware or graphical display.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from attendance_core import (
    load_student_records,
    save_student_record,
    get_student_mapping,
    identify_face,
    mark_attendance,
    is_attendance_marked,
    load_attendance_records,
    draw_face_annotations,
    cleanup_partial_registration,
    LBPH_CONFIDENCE_THRESHOLD,
)


class TestMockedCameraAndWorkflow(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.student_csv = os.path.join(self.test_dir, "StudentDetails.csv")
        self.attendance_dir = os.path.join(self.test_dir, "Attendance")
        self.training_img_dir = os.path.join(self.test_dir, "TrainingImage")
        self.training_label_dir = os.path.join(self.test_dir, "TrainingImageLabel")
        os.makedirs(self.attendance_dir, exist_ok=True)
        os.makedirs(self.training_img_dir, exist_ok=True)
        os.makedirs(self.training_label_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mocked_webcam_registration_success(self):
        """Simulate capturing 100 face images from a camera and saving registration."""
        roll = "2023PECEC390"
        name = "Sanjay P"
        serial = 1

        # Mock frame and detector
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock face coordinates (x, y, w, h)
        mock_faces = [(100, 100, 80, 80)]

        # Simulate camera loop taking 100 samples
        sample_count = 0
        for i in range(1, 101):
            sample_count += 1
            img_path = os.path.join(self.training_img_dir, f"{name}.{serial}.{roll}.{sample_count}.jpg")
            # Save cropped dummy grayscale face
            cv2.imwrite(img_path, fake_frame[100:180, 100:180, 0])

        self.assertEqual(sample_count, 100)
        self.assertEqual(len(os.listdir(self.training_img_dir)), 100)

        # Save record
        save_student_record(self.student_csv, serial, roll, name)
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll"], roll)

    def test_mocked_webcam_registration_aborted(self):
        """Simulate user pressing 'q' after 15 images: images are cleaned up and no CSV row written."""
        roll = "2023PECEC390"
        name = "Sanjay P"
        serial = 1

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate only 15 frames captured before user quit
        for i in range(1, 16):
            img_path = os.path.join(self.training_img_dir, f"{name}.{serial}.{roll}.{i}.jpg")
            cv2.imwrite(img_path, fake_frame[100:180, 100:180, 0])

        self.assertEqual(len(os.listdir(self.training_img_dir)), 15)

        # Cancellation triggers cleanup
        cleanup_partial_registration(self.training_img_dir, serial, roll)
        self.assertEqual(len(os.listdir(self.training_img_dir)), 0)

        # Verify no student was registered
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 0)

    def test_mocked_recognition_and_attendance_loop(self):
        """
        Simulate real-time webcam tracking frames:
        Frame 1: Sanjay P recognized -> attendance marked, overlay drawn.
        Frame 2: Sanjay P recognized again -> attendance NOT duplicated, overlay STILL drawn with Sanjay P and 'Attendance already marked today'.
        Frame 3: Unknown person recognized -> overlay drawn as Unknown, no attendance.
        """
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        records = load_student_records(self.student_csv)
        mapping = get_student_mapping(records)

        today_csv = os.path.join(self.attendance_dir, "Attendance_24-10-2026.csv")
        marked_cache = set()

        # --- FRAME 1: Sanjay P appears for the first time ---
        pred_label, pred_dist = 1, 38.5  # Confident match
        student_name, roll_num, is_known = identify_face(pred_label, pred_dist, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertTrue(is_known)
        self.assertEqual(student_name, "Sanjay P")
        self.assertEqual(roll_num, "2023PECEC390")

        # Mark attendance
        marked, status_msg = mark_attendance(today_csv, roll_num, student_name, cached_set=marked_cache)
        self.assertTrue(marked)
        self.assertIn("marked for Sanjay P", status_msg)

        # Draw overlay on frame
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame1, 100, 100, 120, 120, student_name, roll_num, is_known, status_msg)
        self.assertTrue(np.any(frame1 > 0))

        # --- FRAME 2: Sanjay P remains in front of the camera ---
        pred_label2, pred_dist2 = 1, 41.2
        student_name2, roll_num2, is_known2 = identify_face(pred_label2, pred_dist2, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertTrue(is_known2)
        self.assertEqual(student_name2, "Sanjay P")
        self.assertEqual(roll_num2, "2023PECEC390")

        # Check attendance
        is_already_marked = is_attendance_marked(roll_num2, today_csv, marked_cache)
        self.assertTrue(is_already_marked)
        status_msg2 = "Attendance already marked today"

        # Identity MUST NOT disappear! Draw overlay again with Sanjay P
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame2, 100, 100, 120, 120, student_name2, roll_num2, is_known2, status_msg2)
        self.assertTrue(np.any(frame2 > 0))

        # Check attendance CSV: still exactly 1 record
        att_rows = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows), 1)

        # --- FRAME 3: Unknown person enters frame ---
        pred_label3, pred_dist3 = 1, 88.0  # High distance (unrecognized)
        student_name3, roll_num3, is_known3 = identify_face(pred_label3, pred_dist3, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertFalse(is_known3)
        self.assertEqual(student_name3, "Unknown")
        self.assertEqual(roll_num3, "")

        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame3, 100, 100, 120, 120, student_name3, roll_num3, is_known3, "")
        self.assertTrue(np.any(frame3 > 0))

        # Attendance file still has 1 record
        att_rows = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows), 1)


if __name__ == "__main__":
    unittest.main()
