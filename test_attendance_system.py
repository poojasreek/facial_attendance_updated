"""
test_attendance_system.py
Comprehensive automated test suite covering all requirements from Section 7:
1. Loading registered-student list from persistent records.
2. Correct count and display of student names and roll numbers.
3. Refreshing the list after a successful registration.
4. Empty and malformed student-data files.
5. Successful registration of a unique roll number.
6. Duplicate roll-number rejection.
7. Duplicate detection after trimming whitespace and applying documented case policy.
8. Preservation of existing records after rejected registration.
9. Attendance deduplication for the same student and date.
10. Continued identity display when attendance already exists.
11. Correct mapping from model labels to student names and roll numbers.
12. Unknown-face handling.
13. Failure during image capture or model training without leaving inconsistent registration data.
"""

import os
import csv
import shutil
import tempfile
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from attendance_core import (
    normalize_roll_number,
    validate_student_inputs,
    load_student_records,
    get_student_mapping,
    check_duplicate_roll,
    get_next_serial,
    save_student_record,
    audit_and_report_duplicates,
    get_attendance_filepath,
    load_attendance_records,
    is_attendance_marked,
    mark_attendance,
    identify_face,
    draw_face_annotations,
    cleanup_partial_registration,
    train_lbph_model,
    LBPH_CONFIDENCE_THRESHOLD,
)


class TestAttendanceSystem(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.student_csv = os.path.join(self.test_dir, "StudentDetails.csv")
        self.attendance_dir = os.path.join(self.test_dir, "Attendance")
        self.training_img_dir = os.path.join(self.test_dir, "TrainingImage")
        self.training_label_dir = os.path.join(self.test_dir, "TrainingImageLabel")
        os.makedirs(self.attendance_dir, exist_ok=True)
        os.makedirs(self.training_img_dir, exist_ok=True)
        os.makedirs(self.training_label_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Requirement 1 & 2: Loading registered students & correct count/display
    # --------------------------------------------------------------------------
    def test_01_load_registered_students_and_count(self):
        """Test loading persistent records and ensuring all 25 students display with names and roll numbers."""
        # Create a file with 25 unique students in the original spacer format
        columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
        rows = []
        for i in range(1, 26):
            rows.append([i, '', f'2023PECEC{100 + i}', '', f'Student Name {i}'])

        with open(self.student_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 25, "Must display all 25 unique registered students, not just a count.")
        self.assertEqual(records[0]["roll"], "2023PECEC101")
        self.assertEqual(records[0]["name"], "Student Name 1")
        self.assertEqual(records[24]["roll"], "2023PECEC125")
        self.assertEqual(records[24]["name"], "Student Name 25")

    # --------------------------------------------------------------------------
    # Requirement 3: Refreshing list after a successful registration
    # --------------------------------------------------------------------------
    def test_02_refresh_list_after_registration(self):
        """Test that adding a new student updates the persistent source and is immediately visible on refresh."""
        initial_records = load_student_records(self.student_csv)
        self.assertEqual(len(initial_records), 0)

        # Register first student
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        refreshed_1 = load_student_records(self.student_csv)
        self.assertEqual(len(refreshed_1), 1)
        self.assertEqual(refreshed_1[0]["roll"], "2023PECEC390")
        self.assertEqual(refreshed_1[0]["name"], "Sanjay P")

        # Register second student
        next_serial = get_next_serial(refreshed_1)
        save_student_record(self.student_csv, next_serial, "2023PECEC391", "Pooja K")
        refreshed_2 = load_student_records(self.student_csv)
        self.assertEqual(len(refreshed_2), 2)
        self.assertEqual(refreshed_2[1]["roll"], "2023PECEC391")
        self.assertEqual(refreshed_2[1]["name"], "Pooja K")

    # --------------------------------------------------------------------------
    # Requirement 4: Empty and malformed student-data files
    # --------------------------------------------------------------------------
    def test_03_empty_and_malformed_student_files(self):
        """Gracefully handle missing files, empty files, and corrupted/malformed CSV lines."""
        # Non-existent file
        missing_csv = os.path.join(self.test_dir, "nonexistent.csv")
        self.assertEqual(load_student_records(missing_csv), [])

        # Completely empty file
        with open(self.student_csv, 'w', encoding='utf-8') as f:
            pass
        self.assertEqual(load_student_records(self.student_csv), [])

        # Malformed lines and blank lines
        with open(self.student_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
            writer.writerow([])  # Blank row
            writer.writerow(['not_a_number', '', 'ROLL1', '', 'Name One'])  # Bad serial
            writer.writerow(['1', '', 'ROLL101', '', 'Valid Student'])       # Good row
            writer.writerow(['2', 'only_two_columns'])                       # Incomplete row
            writer.writerow(['3', '', '', '', 'Missing Roll'])               # Missing roll
            writer.writerow(['4', '', 'ROLL104', '', 'Second Valid'])        # Good row

        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 2, "Only valid records should be parsed, corrupt rows safely skipped.")
        self.assertEqual(records[0]["roll"], "ROLL101")
        self.assertEqual(records[1]["roll"], "ROLL104")

    # --------------------------------------------------------------------------
    # Requirement 5: Successful registration of unique roll number
    # --------------------------------------------------------------------------
    def test_04_successful_registration_of_unique_roll(self):
        """Test successful registration of a unique student roll number."""
        valid, msg = validate_student_inputs("2023PECEC390", "Sanjay P")
        self.assertTrue(valid)
        self.assertEqual(msg, "")

        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["serial"], 1)
        self.assertEqual(records[0]["roll"], "2023PECEC390")
        self.assertEqual(records[0]["name"], "Sanjay P")

    # --------------------------------------------------------------------------
    # Requirement 6: Duplicate roll-number rejection
    # --------------------------------------------------------------------------
    def test_05_duplicate_roll_number_rejection(self):
        """Reject duplicate roll number and verify clear error message."""
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        records = load_student_records(self.student_csv)

        is_dup, dup_rec = check_duplicate_roll("2023PECEC390", records)
        self.assertTrue(is_dup)
        self.assertIsNotNone(dup_rec)
        self.assertEqual(dup_rec["name"], "Sanjay P")

        with self.assertRaises(ValueError) as ctx:
            save_student_record(self.student_csv, 2, "2023PECEC390", "Another Name")
        self.assertIn("already exists", str(ctx.exception))

    # --------------------------------------------------------------------------
    # Requirement 7: Duplicate detection after whitespace and case policy
    # --------------------------------------------------------------------------
    def test_06_duplicate_detection_whitespace_and_case(self):
        """Verify normalization catches duplicate with leading/trailing spaces and lowercase/mixed-case."""
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        records = load_student_records(self.student_csv)

        variants = [
            "  2023PECEC390  ",
            "2023pecec390",
            "  2023Pecec390  ",
            "\t2023pecec390\n",
        ]
        for variant in variants:
            is_dup, _ = check_duplicate_roll(variant, records)
            self.assertTrue(is_dup, f"Variant '{variant}' must be detected as duplicate.")

            with self.assertRaises(ValueError):
                save_student_record(self.student_csv, 2, variant, "Imposter")

    # --------------------------------------------------------------------------
    # Requirement 8: Preservation of existing records after rejected registration
    # --------------------------------------------------------------------------
    def test_07_preservation_of_existing_records_after_rejection(self):
        """Ensure original student details remain completely unchanged after duplicate attempt."""
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        
        try:
            save_student_record(self.student_csv, 2, "2023PECEC390", "Overwritten Name")
        except ValueError:
            pass

        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["serial"], 1)
        self.assertEqual(records[0]["roll"], "2023PECEC390")
        self.assertEqual(records[0]["name"], "Sanjay P")

    # --------------------------------------------------------------------------
    # Requirement 9: Attendance deduplication for the same student and date
    # --------------------------------------------------------------------------
    def test_08_attendance_deduplication_same_student_and_date(self):
        """Ensure attendance is written only once per student per date."""
        att_file = os.path.join(self.attendance_dir, "Attendance_24-10-2026.csv")
        marked_cache = set()

        # First mark: Should succeed
        success1, msg1 = mark_attendance(att_file, "2023PECEC390", "Sanjay P", cached_set=marked_cache)
        self.assertTrue(success1)
        self.assertIn("marked for Sanjay P", msg1)

        # Second mark on same day: Must be deduplicated
        success2, msg2 = mark_attendance(att_file, "2023PECEC390", "Sanjay P", cached_set=marked_cache)
        self.assertFalse(success2)
        self.assertEqual(msg2, "Attendance already marked today")

        # Third mark with case/whitespace variations: Must also be deduplicated
        success3, msg3 = mark_attendance(att_file, "  2023pecec390  ", "Sanjay P", cached_set=marked_cache)
        self.assertFalse(success3)
        self.assertEqual(msg3, "Attendance already marked today")

        # Verify physical CSV rows count
        att_records = load_attendance_records(att_file)
        self.assertEqual(len(att_records), 1, "Only a single attendance entry must exist in the file.")

    # --------------------------------------------------------------------------
    # Requirement 10: Continued identity display when attendance already exists
    # --------------------------------------------------------------------------
    def test_09_continued_identity_display_when_attendance_already_exists(self):
        """When face is recognized, identity must always resolve and overlay must display name and roll number."""
        student_mapping = {
            1: {"roll": "2023PECEC390", "name": "Sanjay P", "normalized_roll": "2023PECEC390"}
        }
        att_file = os.path.join(self.attendance_dir, "Attendance_24-10-2026.csv")
        marked_cache = set()

        # Mark attendance first
        mark_attendance(att_file, "2023PECEC390", "Sanjay P", cached_set=marked_cache)
        self.assertTrue(is_attendance_marked("2023PECEC390", att_file, marked_cache))

        # Recognizer recognizes the student (serial=1, distance=35 < 70)
        name, roll, is_known = identify_face(1, 35.0, student_mapping)
        self.assertTrue(is_known)
        self.assertEqual(name, "Sanjay P")
        self.assertEqual(roll, "2023PECEC390")

        # Verify that overlay drawing continues to render the identity and status
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(
            dummy_frame, 100, 100, 150, 150,
            name=name, roll=roll, is_known=True,
            status_msg="Attendance already marked today"
        )
        # Verify the frame was modified (not all zeros)
        self.assertTrue(np.any(dummy_frame > 0), "Overlay text and rectangle must be drawn on frame.")

    # --------------------------------------------------------------------------
    # Requirement 11: Correct mapping from model labels to names and roll numbers
    # --------------------------------------------------------------------------
    def test_10_model_label_mapping(self):
        """Test model label (serial int) correctly resolves to student name and roll number."""
        save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
        save_student_record(self.student_csv, 2, "2023PECEC391", "Pooja K")

        records = load_student_records(self.student_csv)
        mapping = get_student_mapping(records)

        self.assertIn(1, mapping)
        self.assertEqual(mapping[1]["name"], "Sanjay P")
        self.assertEqual(mapping[1]["roll"], "2023PECEC390")

        self.assertIn(2, mapping)
        self.assertEqual(mapping[2]["name"], "Pooja K")
        self.assertEqual(mapping[2]["roll"], "2023PECEC391")

        name, roll, is_known = identify_face(2, 40.0, mapping)
        self.assertTrue(is_known)
        self.assertEqual(name, "Pooja K")
        self.assertEqual(roll, "2023PECEC391")

    # --------------------------------------------------------------------------
    # Requirement 12: Unknown face handling
    # --------------------------------------------------------------------------
    def test_11_unknown_face_handling(self):
        """Test that distance >= threshold or unseen label is identified as Unknown."""
        mapping = {1: {"name": "Sanjay P", "roll": "2023PECEC390"}}

        # Case A: Label exists but distance is high (e.g. 85 >= 70)
        name, roll, is_known = identify_face(1, 85.0, mapping, threshold=LBPH_CONFIDENCE_THRESHOLD)
        self.assertFalse(is_known)
        self.assertEqual(name, "Unknown")
        self.assertEqual(roll, "")

        # Case B: Unknown label (e.g. 99)
        name2, roll2, is_known2 = identify_face(99, 30.0, mapping, threshold=LBPH_CONFIDENCE_THRESHOLD)
        self.assertFalse(is_known2)
        self.assertEqual(name2, "Unknown")

        # Verify overlay draws Unknown in red
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_face_annotations(dummy_frame, 50, 50, 100, 100, "Unknown", "", False)
        self.assertTrue(np.any(dummy_frame > 0))

    # --------------------------------------------------------------------------
    # Requirement 13: Failure handling during image capture & cleanup
    # --------------------------------------------------------------------------
    def test_12_failure_cleanup_and_consistency(self):
        """Test that interrupted/aborted capture removes orphaned images and keeps records clean."""
        serial = 5
        roll = "2023PECEC500"
        name = "Incomplete Student"

        # Simulate 10 partial images created before capture was aborted
        for i in range(1, 11):
            img_name = f"{name}.{serial}.{roll}.{i}.jpg"
            img_path = os.path.join(self.training_img_dir, img_name)
            with open(img_path, 'wb') as f:
                f.write(b"fake image data")

        self.assertEqual(len(os.listdir(self.training_img_dir)), 10)

        # Cleanup triggered on cancellation/error
        deleted = cleanup_partial_registration(self.training_img_dir, serial, roll)
        self.assertEqual(deleted, 10)
        self.assertEqual(len(os.listdir(self.training_img_dir)), 0)

        # Ensure no student record was written
        records = load_student_records(self.student_csv)
        self.assertEqual(len(records), 0)

    # --------------------------------------------------------------------------
    # Model Training without images handling
    # --------------------------------------------------------------------------
    def test_13_train_model_empty_handling(self):
        """Model training gracefully reports when no images exist without crashing."""
        success, msg, count = train_lbph_model(
            training_img_dir=self.training_img_dir,
            model_save_path=os.path.join(self.training_label_dir, "Trainner.yml")
        )
        self.assertFalse(success)
        self.assertIn("No valid training images", msg)
        self.assertEqual(count, 0)

    # --------------------------------------------------------------------------
    # Audit duplicates check
    # --------------------------------------------------------------------------
    def test_14_audit_and_report_duplicates(self):
        """Audit function correctly reports duplicate records in existing CSV."""
        columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
        rows = [
            [1, '', '2023PECEC390', '', 'Sanjay P'],
            [2, '', '2023PECEC391', '', 'Pooja K'],
            [3, '', ' 2023pecec390 ', '', 'Sanjay P Dup'],  # Duplicate
        ]
        with open(self.student_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

        audit = audit_and_report_duplicates(self.student_csv)
        self.assertTrue(audit["has_duplicates"])
        self.assertEqual(audit["total_records"], 3)
        self.assertEqual(audit["unique_students"], 2)
        self.assertIn("2023PECEC390", audit["duplicates"])


if __name__ == "__main__":
    unittest.main()
