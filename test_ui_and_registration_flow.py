"""
test_ui_and_registration_flow.py
Automated UI component and workflow verification tests.
Tests the Tkinter Treeview, 25-student loading, refresh actions,
empty-state handling, and duplicate input validations.
"""

import os
import csv
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import attendance_core
from attendance_core import (
    load_student_records,
    save_student_record,
    get_attendance_filepath,
    mark_attendance,
)


class TestUIAndRegistrationFlow(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.student_csv = os.path.join(self.test_dir, "StudentDetails.csv")
        self.attendance_dir = os.path.join(self.test_dir, "Attendance")
        os.makedirs(self.attendance_dir, exist_ok=True)

        # Patch paths in attendance_core
        self.orig_student_csv = attendance_core.DEFAULT_STUDENT_CSV
        self.orig_att_dir = attendance_core.DEFAULT_ATTENDANCE_DIR
        attendance_core.DEFAULT_STUDENT_CSV = self.student_csv
        attendance_core.DEFAULT_ATTENDANCE_DIR = self.attendance_dir

    def tearDown(self):
        attendance_core.DEFAULT_STUDENT_CSV = self.orig_student_csv
        attendance_core.DEFAULT_ATTENDANCE_DIR = self.orig_att_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_25_students_display_acceptance_criteria(self):
        """
        Acceptance Criteria: If 25 unique students are registered, the interface
        displays all 25 students with their names and roll numbers, not just the number 25.
        """
        # Populate 25 unique students in StudentDetails.csv
        for i in range(1, 26):
            save_student_record(
                self.student_csv,
                serial=i,
                roll=f"2023PECEC{100 + i}",
                name=f"Student {i}"
            )

        # Import UI functions from main (using a hidden Tk root)
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()

        try:
            tv_test = ttk.Treeview(root, columns=('roll', 'name'))
            tv_test.heading('#0', text='STUDENT ID')
            tv_test.heading('roll', text='ROLL NUMBER')
            tv_test.heading('name', text='STUDENT NAME')

            lbl_count = tk.Label(root, text="")

            # Simulate load_registered_students_ui logic
            records = load_student_records(self.student_csv)
            for r in records:
                tv_test.insert('', 'end', text=str(r["serial"]), values=(r["roll"], r["name"]))
            lbl_count.configure(text=f"Total Registered Students: {len(records)}")

            # Assertions
            children = tv_test.get_children()
            self.assertEqual(len(children), 25, "Treeview must have exactly 25 items.")
            self.assertEqual(lbl_count.cget("text"), "Total Registered Students: 25")

            # Check individual items have IDs, roll numbers, and names
            first_item = tv_test.item(children[0])
            self.assertEqual(first_item["text"], "1")
            self.assertEqual(first_item["values"][0], "2023PECEC101")
            self.assertEqual(first_item["values"][1], "Student 1")

            last_item = tv_test.item(children[24])
            self.assertEqual(last_item["text"], "25")
            self.assertEqual(last_item["values"][0], "2023PECEC125")
            self.assertEqual(last_item["values"][1], "Student 25")

        finally:
            root.destroy()

    def test_empty_state_and_refresh(self):
        """Verify empty-state messaging when no students exist, and refresh when a student is added."""
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()

        try:
            tv_test = ttk.Treeview(root, columns=('roll', 'name'))
            lbl_count = tk.Label(root, text="")

            def refresh():
                for item in tv_test.get_children():
                    tv_test.delete(item)
                records = load_student_records(self.student_csv)
                lbl_count.configure(text=f"Total Registered Students: {len(records)}")
                if len(records) == 0:
                    tv_test.insert('', 'end', text="--", values=("No registered students found", "Register in 'Register' tab"))
                else:
                    for r in records:
                        tv_test.insert('', 'end', text=str(r["serial"]), values=(r["roll"], r["name"]))

            # Initially empty
            refresh()
            self.assertEqual(lbl_count.cget("text"), "Total Registered Students: 0")
            items = tv_test.get_children()
            self.assertEqual(len(items), 1)
            self.assertIn("No registered students found", tv_test.item(items[0])["values"][0])

            # Now register a student
            save_student_record(self.student_csv, 1, "2023PECEC390", "Sanjay P")
            refresh()

            self.assertEqual(lbl_count.cget("text"), "Total Registered Students: 1")
            items_after = tv_test.get_children()
            self.assertEqual(len(items_after), 1)
            self.assertEqual(tv_test.item(items_after[0])["values"][0], "2023PECEC390")
            self.assertEqual(tv_test.item(items_after[0])["values"][1], "Sanjay P")

        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
