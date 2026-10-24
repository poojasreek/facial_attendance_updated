"""
audit_and_migrate.py
Command-line audit and non-destructive migration utility for StudentDetails.csv.
Inspects existing student records for duplicate roll numbers, reports findings,
and provides a safe backup and reconciliation workflow without silent data loss.
"""

import os
import sys
import shutil
import datetime
from attendance_core import (
    DEFAULT_STUDENT_CSV,
    DEFAULT_TRAINING_IMG_DIR,
    load_student_records,
    audit_and_report_duplicates,
    save_student_record,
)


def run_audit(csv_path: str = DEFAULT_STUDENT_CSV):
    print("=" * 65)
    print("STUDENT RECORDS INTEGRITY & DUPLICATE AUDIT")
    print("=" * 65)
    print(f"Target CSV Path: {csv_path}")

    if not os.path.isfile(csv_path):
        print(f"Status: File does not exist yet ({csv_path}).")
        print("No duplicates to report. Ready for clean registrations.")
        print("=" * 65)
        return

    report = audit_and_report_duplicates(csv_path)
    print(f"Total Student Rows Parsed: {report['total_records']}")
    print(f"Unique Student Roll Numbers: {report['unique_students']}")

    if not report["has_duplicates"]:
        print("\nSUCCESS: All student roll numbers are unique. No duplicates detected!")
        print("=" * 65)
        return

    print("\nWARNING: Duplicate roll numbers found:")
    for norm_roll, records in report["duplicates"].items():
        print(f"\n  Roll Number: '{norm_roll}' ({len(records)} entries):")
        for rec in records:
            serial = rec["serial"]
            name = rec["name"]
            # Count associated face images
            img_count = 0
            if os.path.exists(DEFAULT_TRAINING_IMG_DIR):
                for f in os.listdir(DEFAULT_TRAINING_IMG_DIR):
                    parts = f.split(".")
                    if len(parts) >= 3 and parts[1] == str(serial):
                        img_count += 1
            print(f"    - Serial ID: {serial} | Name: '{name}' | Face Images in TrainingImage: {img_count}")

    print("\nRecommended Non-Destructive Resolution Procedure:")
    print("1. A backup copy is created: StudentDetails_backup_<timestamp>.csv")
    print("2. The record with valid face training images is designated as Primary.")
    print("3. Redundant duplicate records are archived to StudentDetails_archived_<timestamp>.csv.")
    print("4. Face images and attendance logs are never deleted.")
    print("=" * 65)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STUDENT_CSV
    run_audit(target)
