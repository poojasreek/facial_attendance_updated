"""
attendance_core.py
Core business logic, data persistence, face recognition mapping,
and validation for the Face Recognition Based Attendance System.
"""

import os
import csv
import cv2
import time
import datetime
import tempfile
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Set

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_STUDENT_DIR = "StudentDetails"
DEFAULT_STUDENT_CSV = os.path.join(DEFAULT_STUDENT_DIR, "StudentDetails.csv")
DEFAULT_TRAINING_IMG_DIR = "TrainingImage"
DEFAULT_TRAINING_LABEL_DIR = "TrainingImageLabel"
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_TRAINING_LABEL_DIR, "Trainner.yml")
DEFAULT_PASSWORD_FILE = os.path.join(DEFAULT_TRAINING_LABEL_DIR, "psd.txt")
DEFAULT_ATTENDANCE_DIR = "Attendance"

# LBPH Face Recognizer distance threshold:
# In OpenCV LBPH, predict() returns (label, distance).
# Lower distance indicates higher confidence (0 = exact match).
# Distances below 65-70 represent high-confidence matches.
# Distances >= 70 are categorized as Unknown faces.
LBPH_CONFIDENCE_THRESHOLD = 70.0

# Number of face samples to capture per student registration
SAMPLES_PER_STUDENT = 15



def assure_path_exists(path: str) -> None:
    """Ensure that the parent directory of path exists."""
    dirname = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


# ==============================================================================
# ROLL NUMBER NORMALIZATION & VALIDATION
# ==============================================================================

def normalize_roll_number(roll: str) -> str:
    """
    Documented Normalization Policy:
    Roll numbers are case-insensitive and invariant to surrounding whitespace.
    Leading and trailing whitespace is stripped, internal multiple spaces are collapsed,
    and all ASCII letters are converted to UPPERCASE.
    Example: '  2023pecec390  ' -> '2023PECEC390'
    """
    if not roll:
        return ""
    stripped = roll.strip()
    # Normalize internal spaces and convert to uppercase
    normalized = " ".join(stripped.split()).upper()
    return normalized


def validate_student_inputs(roll: str, name: str) -> Tuple[bool, str]:
    """
    Validates user input for student registration.
    Returns (is_valid, error_message).
    """
    if not roll or not roll.strip():
        return False, "Roll number cannot be empty."
    
    clean_roll = roll.strip()
    # Allowed roll number characters: alphanumeric, hyphen, underscore, slash, dot
    valid_roll_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.")
    if not all(c in valid_roll_chars for c in clean_roll):
        return False, "Roll number contains invalid characters. Use letters, numbers, hyphens, underscores, or slashes."

    if not name or not name.strip():
        return False, "Student name cannot be empty."
    
    clean_name = name.strip()
    # Name must contain at least one alphabetic character
    if not any(c.isalpha() for c in clean_name):
        return False, "Student name must contain letters."
    
    # Allowed name characters: letters, spaces, dots, hyphens, apostrophes
    valid_name_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'-")
    if not all(c in valid_name_chars for c in clean_name):
        return False, "Student name contains invalid characters. Please enter a valid name."
    
    return True, ""


# ==============================================================================
# STUDENT CSV MANAGEMENT & DATA INTEGRITY
# ==============================================================================

def parse_student_row(raw_row: List[str]) -> Optional[Dict]:
    """
    Parses a single row from StudentDetails.csv.
    Handles both original format with empty column spacers:
      ['SERIAL NO.', '', 'ID', '', 'NAME'] -> [serial, '', roll, '', name]
    and standard CSV format:
      ['SERIAL NO.', 'ID', 'NAME'] -> [serial, roll, name]
    """
    # Filter out empty spacer items if it is the spacer format
    filtered = [col.strip() for col in raw_row if col is not None]
    non_empty = [c for c in filtered if c != ""]
    
    if len(non_empty) < 3:
        return None
    
    # Check if first item is header
    first_col = non_empty[0].upper()
    if "SERIAL" in first_col or "ID" in first_col or "ROLL" in first_col:
        # Check if this is the header row
        if any("NAME" in c.upper() for c in non_empty):
            return None
    
    try:
        serial = int(non_empty[0])
    except ValueError:
        return None
    
    roll = non_empty[1].strip()
    name = non_empty[2].strip()
    
    if not roll or not name:
        return None
        
    return {
        "serial": serial,
        "roll": roll,
        "name": name,
        "normalized_roll": normalize_roll_number(roll)
    }


def load_student_records(csv_path: str = DEFAULT_STUDENT_CSV) -> List[Dict]:
    """
    Reads all student records from the persistent CSV file.
    Gracefully handles:
    - Missing files (returns empty list)
    - Empty files (returns empty list)
    - Extra blank rows (e.g. from Windows \r\r\n)
    - Malformed rows (skips corrupted rows safely)
    """
    if not os.path.isfile(csv_path):
        return []
    
    records = []
    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                parsed = parse_student_row(row)
                if parsed:
                    records.append(parsed)
    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}")
        return []
        
    return records


def get_student_mapping(records: List[Dict]) -> Dict[int, Dict[str, str]]:
    """
    Returns a dictionary mapping LBPH model integer label (serial) to student info:
    {serial: {'roll': roll, 'name': name, 'normalized_roll': normalized_roll}}
    """
    mapping = {}
    for r in records:
        mapping[r["serial"]] = {
            "roll": r["roll"],
            "name": r["name"],
            "normalized_roll": r["normalized_roll"]
        }
    return mapping


def check_duplicate_roll(roll: str, records: List[Dict]) -> Tuple[bool, Optional[Dict]]:
    """
    Checks if a normalized roll number already exists in student records.
    Returns (True, matching_record) if found, else (False, None).
    """
    norm_target = normalize_roll_number(roll)
    for r in records:
        if r["normalized_roll"] == norm_target:
            return True, r
    return False, None


def get_next_serial(records: List[Dict]) -> int:
    """
    Computes the next integer serial number for LBPH training.
    Uses max existing serial + 1 to avoid label collisions.
    """
    if not records:
        return 1
    return max(r["serial"] for r in records) + 1


def save_student_record(csv_path: str, serial: int, roll: str, name: str) -> None:
    """
    Atomically writes a new student record to StudentDetails.csv.
    Rechecks uniqueness immediately before write to avoid race conditions.
    Preserves original column spacer format: ['SERIAL NO.', '', 'ID', '', 'NAME']
    """
    assure_path_exists(csv_path)
    
    # Authoritative disk re-check
    existing_records = load_student_records(csv_path)
    is_dup, dup_record = check_duplicate_roll(roll, existing_records)
    if is_dup:
        raise ValueError(f"Roll number {roll} already exists. Please enter a unique roll number.")
    
    dir_name = os.path.dirname(csv_path) or "."
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    
    # Write to temp file in the same directory for atomic replace
    fd, tmp_file = tempfile.mkstemp(dir=dir_name, prefix="student_details_", suffix=".tmp")
    os.close(fd)
    
    try:
        with open(tmp_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            # Write existing records
            for r in existing_records:
                writer.writerow([r["serial"], '', r["roll"], '', r["name"]])
            # Append new record
            clean_roll = roll.strip()
            clean_name = name.strip()
            writer.writerow([serial, '', clean_roll, '', clean_name])
            f.flush()
            os.fsync(f.fileno())
            
        # Atomic replace
        os.replace(tmp_file, csv_path)
    except Exception as e:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        raise e


def audit_and_report_duplicates(csv_path: str = DEFAULT_STUDENT_CSV) -> Dict:
    """
    Inspects existing student CSV file for duplicate roll numbers.
    Does not delete or modify data; returns a detailed diagnostic report.
    """
    records = load_student_records(csv_path)
    seen: Dict[str, List[Dict]] = {}
    for r in records:
        norm = r["normalized_roll"]
        if norm not in seen:
            seen[norm] = []
        seen[norm].append(r)
    
    duplicates = {norm: recs for norm, recs in seen.items() if len(recs) > 1}
    return {
        "total_records": len(records),
        "unique_students": len(seen),
        "has_duplicates": len(duplicates) > 0,
        "duplicates": duplicates
    }


# ==============================================================================
# ATTENDANCE RECORDING & DEDUPLICATION
# ==============================================================================

def get_attendance_filepath(attendance_dir: str = DEFAULT_ATTENDANCE_DIR, dt: Optional[datetime.date] = None) -> str:
    """Returns the attendance CSV path for the given date (default: today)."""
    if dt is None:
        dt = datetime.date.today()
    date_str = dt.strftime('%d-%m-%Y')
    return os.path.join(attendance_dir, f"Attendance_{date_str}.csv")


def load_attendance_records(filepath: str) -> List[Dict]:
    """
    Loads all attendance records for a specific date file.
    Preserves original format: ['Id', '', 'Name', '', 'Date', '', 'Time']
    """
    if not os.path.isfile(filepath):
        return []
    
    records = []
    try:
        with open(filepath, mode="r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                filtered = [col.strip() for col in row if col is not None]
                non_empty = [c for c in filtered if c != ""]
                if len(non_empty) < 4:
                    continue
                # Skip header
                if "ID" in non_empty[0].upper() and "NAME" in non_empty[1].upper():
                    continue
                roll = non_empty[0]
                name = non_empty[1]
                date_val = non_empty[2]
                time_val = non_empty[3]
                records.append({
                    "roll": roll,
                    "normalized_roll": normalize_roll_number(roll),
                    "name": name,
                    "date": date_val,
                    "time": time_val
                })
    except Exception as e:
        print(f"Warning: Error reading attendance file {filepath}: {e}")
        return []
    return records


def is_attendance_marked(roll: str, filepath: str, cached_set: Optional[Set[str]] = None) -> bool:
    """
    Checks whether attendance has already been recorded for the student today.
    Uses memory cache if provided, and verifies against persistent file.
    """
    norm_roll = normalize_roll_number(roll)
    if cached_set is not None and norm_roll in cached_set:
        return True
    
    records = load_attendance_records(filepath)
    for r in records:
        if r["normalized_roll"] == norm_roll:
            if cached_set is not None:
                cached_set.add(norm_roll)
            return True
            
    return False


def mark_attendance(filepath: str, roll: str, name: str,
                    dt: Optional[datetime.datetime] = None,
                    cached_set: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """
    Records attendance if not already marked today.
    Returns (True, message) if marked, or (False, "Attendance already marked today") if duplicate.
    Maintains compatibility with existing CSV schema: ['Id', '', 'Name', '', 'Date', '', 'Time'].
    """
    assure_path_exists(filepath)
    if is_attendance_marked(roll, filepath, cached_set):
        return False, "Attendance already marked today"
    
    if dt is None:
        dt = datetime.datetime.now()
    
    date_str = dt.strftime('%d-%m-%Y')
    time_str = dt.strftime('%H:%M:%S')
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    row_data = [str(roll), '', str(name), '', str(date_str), '', str(time_str)]
    
    file_exists = os.path.isfile(filepath)
    try:
        with open(filepath, mode="a+", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(col_names)
            writer.writerow(row_data)
            f.flush()
    except Exception as e:
        return False, f"Error saving attendance: {e}"
        
    if cached_set is not None:
        cached_set.add(normalize_roll_number(roll))
        
    return True, f"Attendance marked for {name}"


# ==============================================================================
# FACE RECOGNITION IDENTIFICATION & OVERLAY DRAWING
# ==============================================================================

def identify_face(predicted_serial: int,
                  distance: float,
                  student_mapping: Dict[int, Dict[str, str]],
                  threshold: float = LBPH_CONFIDENCE_THRESHOLD) -> Tuple[str, str, bool]:
    """
    Separates identity resolution from attendance logic.
    Given LBPH prediction (predicted_serial, distance):
    - If distance < threshold and predicted_serial is in registered student mapping:
      returns (student_name, roll_number, True)
    - Otherwise:
      returns ("Unknown", "", False)
    """
    if distance < threshold and predicted_serial in student_mapping:
        info = student_mapping[predicted_serial]
        return info["name"], info["roll"], True
    return "Unknown", "", False


def draw_face_annotations(frame: np.ndarray,
                          x: int, y: int, w: int, h: int,
                          name: str,
                          roll: str,
                          is_known: bool,
                          status_msg: str = "") -> None:
    """
    Draws bounding box, identity label (Name and Roll Number), and attendance status
    directly above or adjacent to the detected face.
    Ensures high visibility with background rectangle to prevent text from blending
    into the camera background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    
    if is_known:
        box_color = (0, 220, 0)      # Bright Green for known
        header_text = f"{name} ({roll})" if roll else name
    else:
        box_color = (0, 0, 255)      # Red for unknown
        header_text = "Unknown"
        status_msg = ""
        
    # Draw face bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    
    # Calculate text sizes
    (tw_head, th_head), baseline_head = cv2.getTextSize(header_text, font, font_scale, thickness)
    tw_status, th_status = (0, 0)
    if status_msg:
        (tw_status, th_status), _ = cv2.getTextSize(status_msg, font, 0.45, 1)
        
    total_height = th_head + (th_status + 6 if status_msg else 0) + 12
    total_width = max(tw_head, tw_status) + 16
    
    # Position above the box if room, else inside/adjacent
    box_y2 = y - 6
    box_y1 = box_y2 - total_height
    if box_y1 < 0:
        box_y1 = y + h + 6
        box_y2 = box_y1 + total_height
        
    box_x1 = max(0, x)
    box_x2 = min(frame.shape[1], box_x1 + total_width)
    
    # Background rectangle for crisp contrast
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 1)
    
    # Identity text
    text_y = box_y1 + th_head + 6
    text_color = (255, 255, 255)
    cv2.putText(frame, header_text, (box_x1 + 8, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Status text (e.g. "Attendance already marked today" or "Attendance Marked")
    if status_msg:
        status_y = text_y + th_status + 6
        status_color = (0, 255, 255) if "already" in status_msg.lower() else (0, 255, 100)
        cv2.putText(frame, status_msg, (box_x1 + 8, status_y), font, 0.45, status_color, 1, cv2.LINE_AA)


# ==============================================================================
# CLEANUP & MAINTENANCE UTILITIES
# ==============================================================================

def cleanup_partial_registration(training_dir: str, serial: int, roll: str) -> int:
    """
    Safely cleans up any captured face images if registration fails,
    is cancelled, or produces fewer than required images.
    Prevents orphaned training images from corrupting future model training.
    Returns the count of deleted images.
    """
    if not os.path.exists(training_dir):
        return 0
    
    deleted_count = 0
    norm_roll = normalize_roll_number(roll)
    for fname in os.listdir(training_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        # Format is: [name].[serial].[roll].[sampleNum].jpg
        parts = fname.split(".")
        if len(parts) >= 4:
            try:
                f_serial = int(parts[1])
                f_roll = parts[2].strip()
                if f_serial == serial or normalize_roll_number(f_roll) == norm_roll:
                    full_path = os.path.join(training_dir, fname)
                    try:
                        os.remove(full_path)
                        deleted_count += 1
                    except OSError:
                        pass
            except (ValueError, IndexError):
                continue
                
    return deleted_count


# ==============================================================================
# MODEL TRAINING LOGIC
# ==============================================================================

def get_images_and_labels(training_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Loads all training images and extracts face arrays and integer label IDs.
    Returns (faces, ids).
    """
    if not os.path.exists(training_dir):
        return [], []
        
    image_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces = []
    ids = []
    
    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            # Extract serial ID (parts[1])
            parts = os.path.split(image_path)[-1].split(".")
            if len(parts) >= 3:
                label_id = int(parts[1])
                faces.append(image_np)
                ids.append(label_id)
        except Exception as e:
            print(f"Skipping unreadable training image {image_path}: {e}")
            continue
            
    return faces, ids


def train_lbph_model(training_img_dir: str = DEFAULT_TRAINING_IMG_DIR,
                     model_save_path: str = DEFAULT_MODEL_FILE) -> Tuple[bool, str, int]:
    """
    Trains the OpenCV LBPH face recognizer on captured training images.
    Returns (success: bool, status_message: str, unique_student_count: int).
    """
    faces, ids = get_images_and_labels(training_img_dir)
    if not faces or not ids:
        return False, "No valid training images found. Please register someone first!", 0
        
    try:
        # Compatibility check for LBPHFaceRecognizer create method
        if hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            recognizer = cv2.face_LBPHFaceRecognizer.create()
            
        recognizer.train(faces, np.array(ids))
        assure_path_exists(model_save_path)
        recognizer.save(model_save_path)
        unique_students = len(set(ids))
        return True, f"Model trained successfully for {unique_students} student(s)!", unique_students
    except Exception as e:
        return False, f"Model training failed: {e}", 0
