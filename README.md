# Face_recognition_based_attendance_system
A python GUI integrated attendance system using face recognition to take attendance.

In this python project, I have made an attendance system which takes attendance by using face recognition technique. I have also intergrated it with GUI (Graphical user interface) so it can be easy to use by anyone. GUI for this project is also made on python using tkinter.

TECHNOLOGY USED:
1) tkinter for whole GUI
2) OpenCV for taking images and face recognition (cv2.face.LBPHFaceRecognizer_create())
3) CSV, Numpy, Pandas, datetime etc. for other purposes.

FEATURES:
1) Easy to use with interactive GUI support.
2) Password protection for new person registration.
3) Creates/Updates CSV file for details of students on registration.
4) Creates a new CSV file everyday for attendance and marks attendance with proper date and time.
5) Displays live attendance updates for the day on the main screen in tabular format with Id, name, date and time.

### For more information:
https://machinelearningprojects.net/face-recognition-based-attendance-system/

# SCREENSHOTS
MAIN SCREEN:
![Screenshot (9)](https://user-images.githubusercontent.com/37211676/58502148-97ec2a00-81a3-11e9-963e-674b9c3e05dc.png)

TAKING ATTENDANCE:
![Screenshot (10)](https://user-images.githubusercontent.com/37211676/58502149-97ec2a00-81a3-11e9-9658-8968da396c2e.png)

SHOWING ATTENDANCE TAKEN:
![Screenshot (11)](https://user-images.githubusercontent.com/37211676/58502151-9884c080-81a3-11e9-9a90-fec29940ee5a.png)

HELP OPTION IN MENUBAR:
![Screenshot (12)](https://user-images.githubusercontent.com/37211676/58502152-991d5700-81a3-11e9-861a-9115526010c2.png)

CHANGE PASSWORD OPTION:
![Screenshot (13)](https://user-images.githubusercontent.com/37211676/58502146-97539380-81a3-11e9-8536-0c68160ecc55.png)

---

## Repaired Features and Enhancements

### 1. Registered Students Directory (Feature A)
- Full interactive `ttk.Treeview` table displaying **Student ID**, **Roll Number**, and **Student Name**.
- Dynamically populated from authoritative persistent records (`StudentDetails/StudentDetails.csv`).
- Displays the exact registered student count synchronized with the table view.
- Includes a manual **Refresh List** action, automatic reload on tab selection, and graceful empty-state handling.

### 2. Real-Time Identity Display Above Face (Feature B)
- Face identification is completely separated from attendance deduplication.
- Recognizes students and continuously displays their **Name and Roll Number** directly above the bounding box on every frame.
- If attendance has already been recorded today, the overlay continues to show their identity with a status indicator (`Attendance already marked today`) without appending duplicate CSV records.
- LBPH confidence/distance threshold is set to a calibrated `70.0` (distances < 70 indicate high-confidence identification, >= 70 render as distinct red `Unknown` boxes).

### 3. Roll Number Uniqueness Enforcement (Feature C)
- Strict uniqueness policy: Roll numbers are case-insensitive and invariant to surrounding whitespace (e.g., ` 2023pecec390 ` normalizes to `2023PECEC390`).
- Authoritative pre-check against persistent storage before initializing camera or capturing training images.
- Safe atomic file replacement prevents partial writes and file corruption.
- Interrupted or cancelled image captures are safely cleaned up to prevent orphaned images or corrupted dataset states.

### Running the Application & Test Suite

```bash
# Run main application
python main.py

# Run comprehensive automated tests
python run_all_tests.py

# Audit existing student records for duplicate roll numbers
python audit_and_migrate.py
```

