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
    normalize_roll_number,
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
        """Simulate capturing 15 face images from a camera and saving registration."""
        roll = "2023PECEC390"
        name = "Sanjay P"
        serial = 1

        # Mock frame and detector
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate camera loop taking 15 samples
        sample_count = 0
        for i in range(1, 16):
            sample_count += 1
            img_path = os.path.join(self.training_img_dir, f"{name}.{serial}.{roll}.{sample_count}.jpg")
            # Save cropped dummy grayscale face
            cv2.imwrite(img_path, fake_frame[100:180, 100:180, 0])

        self.assertEqual(sample_count, 15)
        self.assertEqual(len(os.listdir(self.training_img_dir)), 15)

        # Save record
        save_student_record(self.student_csv, serial, roll, name)
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["roll"], roll)

    def test_mocked_webcam_registration_aborted(self):
        """Simulate user pressing 'q' after 7 images (less than 15): images are cleaned up and no CSV row written."""
        roll = "2023PECEC390"
        name = "Sanjay P"
        serial = 1

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate only 7 frames captured before user pressed 'q'
        for i in range(1, 8):
            img_path = os.path.join(self.training_img_dir, f"{name}.{serial}.{roll}.{i}.jpg")
            cv2.imwrite(img_path, fake_frame[100:180, 100:180, 0])

        self.assertEqual(len(os.listdir(self.training_img_dir)), 7)

        # Cancellation triggers cleanup
        cleanup_partial_registration(self.training_img_dir, serial, roll)
        self.assertEqual(len(os.listdir(self.training_img_dir)), 0)

        # Verify no student was registered
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 0)

    def test_mocked_recognition_and_attendance_loop(self):
        """
        Simulate real-time webcam tracking frames with the new session-based status logic:

        Session state:
          previously_marked_rolls: rolls marked BEFORE camera opened (pre-loaded from CSV)
          session_marked_rolls:    rolls marked IN this session {norm_roll: time_str}

        Frame 1: Sanjay P first seen this session -> attendance marked, status = 'Attendance Marked! (HH:MM:SS)'
        Frame 2: Sanjay P seen again in same session -> status = 'Attendance Marked! (same time)', no duplicate row
        Frame 3: Unknown person -> overlay drawn as Unknown, no attendance written
        Frame 4: Student pre-marked before session -> status = 'Already marked today (HH:MM:SS)'
        """
        import datetime as dt_module

        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        save_student_record(self.student_csv, 2, "2023PECEC391", "Pooja K")
        records = load_student_records(self.student_csv)
        mapping = get_student_mapping(records)

        today_csv = os.path.join(self.attendance_dir, "Attendance_25-10-2026.csv")

        # Pre-mark Pooja K before camera opens (simulate earlier session or restart)
        mark_attendance(today_csv, "2023PECEC391", "Pooja K")

        # Load pre-session state (mirrors TrackImages startup)
        initial_records = load_attendance_records(today_csv)
        previously_marked_rolls = {r["normalized_roll"]: r.get("time", "") for r in initial_records}
        session_marked_rolls = {}  # newly marked in this camera session

        # Verify Pooja K is pre-marked
        self.assertIn("2023PECEC391", previously_marked_rolls)
        self.assertNotIn("2023PECEC390", previously_marked_rolls)

        # --- FRAME 1: Sanjay P appears for the first time this session ---
        student_name, roll_num, is_known = identify_face(1, 38.5, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertTrue(is_known)
        self.assertEqual(student_name, "Sanjay P")

        norm_roll = normalize_roll_number(roll_num)
        self.assertNotIn(norm_roll, session_marked_rolls)
        self.assertNotIn(norm_roll, previously_marked_rolls)

        now = dt_module.datetime.now()
        time_str = now.strftime('%H:%M:%S')
        marked, _ = mark_attendance(today_csv, roll_num, student_name, dt=now)
        self.assertTrue(marked)
        session_marked_rolls[norm_roll] = time_str
        status_msg1 = f"Attendance Marked! ({time_str})"

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame1, 100, 100, 120, 120, student_name, roll_num, is_known, status_msg1)
        self.assertTrue(np.any(frame1 > 0))

        # Verify exactly 2 rows in CSV (Pooja K + Sanjay P)
        att_rows = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows), 2)

        # --- FRAME 2: Sanjay P still in front of camera ---
        student_name2, roll_num2, is_known2 = identify_face(1, 41.2, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertTrue(is_known2)
        norm_roll2 = normalize_roll_number(roll_num2)

        # Must be in session_marked_rolls (marked this session)
        self.assertIn(norm_roll2, session_marked_rolls)
        status_msg2 = f"Attendance Marked! ({session_marked_rolls[norm_roll2]})"
        self.assertIn("Attendance Marked!", status_msg2)

        # Identity overlay still draws Sanjay P
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame2, 100, 100, 120, 120, student_name2, roll_num2, is_known2, status_msg2)
        self.assertTrue(np.any(frame2 > 0))

        # CSV still has exactly 2 rows - no duplicate written
        att_rows2 = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows2), 2)

        # --- FRAME 3: Unknown person enters frame ---
        student_name3, roll_num3, is_known3 = identify_face(1, 88.0, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertFalse(is_known3)
        self.assertEqual(student_name3, "Unknown")
        self.assertEqual(roll_num3, "")

        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame3, 100, 100, 120, 120, student_name3, roll_num3, is_known3, "")
        self.assertTrue(np.any(frame3 > 0))

        # Still 2 rows
        att_rows3 = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows3), 2)

        # --- FRAME 4: Pooja K (pre-marked before session) enters frame ---
        student_name4, roll_num4, is_known4 = identify_face(2, 35.0, mapping, LBPH_CONFIDENCE_THRESHOLD)
        self.assertTrue(is_known4)
        self.assertEqual(student_name4, "Pooja K")

        norm_roll4 = normalize_roll_number(roll_num4)
        self.assertNotIn(norm_roll4, session_marked_rolls)   # not marked this session
        self.assertIn(norm_roll4, previously_marked_rolls)   # was marked before session opened

        prev_time = previously_marked_rolls[norm_roll4]
        status_msg4 = f"Already marked today ({prev_time})" if prev_time else "Attendance already marked today"
        self.assertIn("marked today", status_msg4)

        # Identity still shown for Pooja K
        frame4 = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(frame4, 100, 100, 120, 120, student_name4, roll_num4, is_known4, status_msg4)
        self.assertTrue(np.any(frame4 > 0))

        # CSV still has exactly 2 rows
        att_rows4 = load_attendance_records(today_csv)
        self.assertEqual(len(att_rows4), 2)


if __name__ == "__main__":
    unittest.main()
