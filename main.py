############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import datetime
import time

# Core business logic and data persistence
from attendance_core import (
    DEFAULT_CASCADE_PATH,
    DEFAULT_STUDENT_DIR,
    DEFAULT_STUDENT_CSV,
    DEFAULT_TRAINING_IMG_DIR,
    DEFAULT_TRAINING_LABEL_DIR,
    DEFAULT_MODEL_FILE,
    DEFAULT_PASSWORD_FILE,
    DEFAULT_ATTENDANCE_DIR,
    LBPH_CONFIDENCE_THRESHOLD,
    SAMPLES_PER_STUDENT,
    assure_path_exists,
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
)

############################################# FUNCTIONS ################################################

def tick():
    """Updates the digital clock on the GUI every 200 milliseconds."""
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


def contact():
    """Displays contact help message box."""
    mess.showinfo(title='Contact us', message="Please contact us on : 'help.attendance.system@gmail.com' ")


def check_haarcascadefile():
    """Checks if the Haar cascade frontal face xml file exists."""
    exists = os.path.isfile(DEFAULT_CASCADE_PATH)
    if not exists:
        mess.showerror(
            title='Haar Cascade Missing',
            message=f"Missing '{DEFAULT_CASCADE_PATH}'. Please ensure it is located in the application directory."
        )
        return False
    return True


def get_stored_password():
    """Reads the stored password or returns None if not set."""
    if os.path.isfile(DEFAULT_PASSWORD_FILE):
        try:
            with open(DEFAULT_PASSWORD_FILE, "r", encoding="utf-8") as tf:
                return tf.read().strip()
        except Exception:
            return None
    return None


def save_pass():
    """Handles password modification and registration."""
    assure_path_exists(DEFAULT_PASSWORD_FILE)
    key = get_stored_password()
    
    if key is None:
        master.destroy()
        new_pas = tsd.askstring('Password Setup', 'Please enter a new password below', show='*')
        if not new_pas:
            mess.showwarning(title='No Password Entered', message='Password not set! Please try again.')
        else:
            with open(DEFAULT_PASSWORD_FILE, "w", encoding="utf-8") as tf:
                tf.write(new_pas.strip())
            mess.showinfo(title='Password Registered', message='New password was registered successfully!')
        return

    op = old.get().strip()
    newp = new.get().strip()
    nnewp = nnew.get().strip()
    
    if op == key:
        if not newp:
            mess.showerror(title='Error', message='New password cannot be empty!')
            return
        if newp == nnewp:
            with open(DEFAULT_PASSWORD_FILE, "w", encoding="utf-8") as txf:
                txf.write(newp)
            mess.showinfo(title='Password Changed', message='Password changed successfully!')
            master.destroy()
        else:
            mess.showerror(title='Error', message='Confirm new password does not match!')
    else:
        mess.showerror(title='Wrong Password', message='Please enter correct old password.')


def change_pass():
    """Opens the Change Password modal dialog."""
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.geometry("400x170")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    
    lbl4 = tk.Label(master, text='Enter Old Password', bg='white', font=('times', 11, ' bold '))
    lbl4.place(x=15, y=15)
    old = tk.Entry(master, width=22, fg="black", relief='solid', font=('times', 11, ' bold '), show='*')
    old.place(x=185, y=15)
    
    lbl5 = tk.Label(master, text='Enter New Password', bg='white', font=('times', 11, ' bold '))
    lbl5.place(x=15, y=50)
    new = tk.Entry(master, width=22, fg="black", relief='solid', font=('times', 11, ' bold '), show='*')
    new.place(x=185, y=50)
    
    lbl6 = tk.Label(master, text='Confirm Password', bg='white', font=('times', 11, ' bold '))
    lbl6.place(x=15, y=85)
    nnew = tk.Entry(master, width=22, fg="black", relief='solid', font=('times', 11, ' bold '), show='*')
    nnew.place(x=185, y=85)
    
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height=1, width=15, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=40, y=125)
    cancel = tk.Button(master, text="Cancel", command=master.destroy, fg="white", bg="#ea2a2a", height=1, width=15, activebackground="white", font=('times', 10, ' bold '))
    cancel.place(x=210, y=125)


def psw():
    """Authenticates password before training the facial recognition model."""
    assure_path_exists(DEFAULT_PASSWORD_FILE)
    key = get_stored_password()
    
    if key is None:
        new_pas = tsd.askstring('Password Setup', 'Please create a master password below', show='*')
        if not new_pas:
            mess.showwarning(title='No Password Entered', message='Password not set! Cannot train without password.')
            return
        with open(DEFAULT_PASSWORD_FILE, "w", encoding="utf-8") as tf:
            tf.write(new_pas.strip())
        mess.showinfo(title='Password Registered', message='New password was registered successfully!')
        TrainImages()
        return

    password = tsd.askstring('Password', 'Enter Password to Train Model', show='*')
    if password is None:
        return
    if password.strip() == key:
        TrainImages()
    else:
        mess.showerror(title='Wrong Password', message='You entered the wrong password!')


def clear():
    """Clears the Roll Number input entry."""
    txt.delete(0, 'end')
    message1.configure(text="1) Take Images  >>>  2) Save Profile")


def clear2():
    """Clears the Student Name input entry."""
    txt2.delete(0, 'end')
    message1.configure(text="1) Take Images  >>>  2) Save Profile")


############################################# REGISTRATION ################################################

def TakeImages():
    """
    Captures 100 face images for a new student registration.
    Enforces roll number uniqueness BEFORE opening camera.
    Validates input fields, handles camera capture cancellation safely,
    and atomically saves student record.
    """
    if not check_haarcascadefile():
        return

    raw_roll = txt.get()
    raw_name = txt2.get()

    # 1. Validate inputs
    is_valid, err_msg = validate_student_inputs(raw_roll, raw_name)
    if not is_valid:
        mess.showerror("Invalid Input", err_msg)
        message1.configure(text=err_msg)
        return

    roll = normalize_roll_number(raw_roll)
    name = raw_name.strip()

    # 2. Enforce Roll Number Uniqueness BEFORE expensive operations
    existing_records = load_student_records(DEFAULT_STUDENT_CSV)
    is_dup, dup_record = check_duplicate_roll(roll, existing_records)
    if is_dup:
        err_dup = f"Roll number {roll} already exists. Please enter a unique roll number."
        mess.showerror("Duplicate Roll Number", err_dup)
        message1.configure(text=err_dup)
        return

    # 3. Determine next serial ID for LBPH recognizer
    serial = get_next_serial(existing_records)
    assure_path_exists(DEFAULT_TRAINING_IMG_DIR)

    # 4. Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess.showerror("Camera Error", "Could not access webcam. Please check your camera connection.")
        return

    detector = cv2.CascadeClassifier(DEFAULT_CASCADE_PATH)
    sample_num = 0
    message1.configure(text=f"Capturing faces for {name}... Press 'q' to Cancel.")
    window.update()

    completed_successfully = False
    try:
        while True:
            ret, img = cam.read()
            if not ret or img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sample_num += 1
                # Format: name.serial.roll.sampleNum.jpg
                img_path = os.path.join(
                    DEFAULT_TRAINING_IMG_DIR,
                    f"{name}.{serial}.{roll}.{sample_num}.jpg"
                )
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    img,
                    f"Sample: {sample_num}/{SAMPLES_PER_STUDENT}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # Informative overlay
            cv2.putText(
                img,
                f"Registering: {name} ({roll}) - Press 'q' to cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.imshow('Taking Images', img)

            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            if sample_num >= SAMPLES_PER_STUDENT:
                completed_successfully = True
                break

            # Handle OpenCV window close 'X' button
            try:
                if cv2.getWindowProperty('Taking Images', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

    finally:
        cam.release()
        cv2.destroyAllWindows()

    # 5. Handle capture outcome
    if not completed_successfully:
        # Clean up partial images to prevent orphaned records
        cleaned = cleanup_partial_registration(DEFAULT_TRAINING_IMG_DIR, serial, roll)
        cancel_msg = f"Registration cancelled. {cleaned} partial images removed."
        message1.configure(text=cancel_msg)
        mess.showwarning("Registration Incomplete", cancel_msg)
        return

    # 6. Recheck uniqueness and save atomically to authoritative CSV
    try:
        save_student_record(DEFAULT_STUDENT_CSV, serial, roll, name)
    except ValueError as e:
        cleanup_partial_registration(DEFAULT_TRAINING_IMG_DIR, serial, roll)
        mess.showerror("Registration Error", str(e))
        message1.configure(text=str(e))
        return
    except Exception as e:
        cleanup_partial_registration(DEFAULT_TRAINING_IMG_DIR, serial, roll)
        mess.showerror("Save Error", f"Failed to save student record: {e}")
        message1.configure(text=f"Failed to save: {e}")
        return

    # 7. Refresh GUI Registered Students list & count
    load_registered_students_ui()
    success_msg = f"Images Taken for Roll: {roll}. Click 'Save Profile' to train model."
    message1.configure(text=success_msg)
    mess.showinfo("Success", f"Registration images captured for {name} ({roll})!\nNow click 'Save Profile' to train the model.")


############################################# MODEL TRAINING ################################################

def TrainImages():
    """Trains the LBPH face recognition model and updates total registration status."""
    if not check_haarcascadefile():
        return

    assure_path_exists(DEFAULT_TRAINING_LABEL_DIR)
    message1.configure(text="Training facial recognition model, please wait...")
    window.update()

    success, msg, unique_students = train_lbph_model(DEFAULT_TRAINING_IMG_DIR, DEFAULT_MODEL_FILE)
    if success:
        message1.configure(text="Profile Saved Successfully")
        load_registered_students_ui()
        mess.showinfo('Training Complete', f"{msg}\nAll registered students are now ready for attendance.")
    else:
        message1.configure(text="Training Failed")
        mess.showwarning('Training Notice', msg)


############################################# ATTENDANCE TRACKING ################################################

def TrackImages():
    """
    Webcam face recognition and attendance recording loop.
    Separates student identification from attendance deduplication.
    Draws identity (Name and Roll Number) above the recognized face on every frame.
    Deduplicates attendance per student per date.
    """
    if not check_haarcascadefile():
        return

    if not os.path.isfile(DEFAULT_MODEL_FILE):
        mess.showerror('Model Missing', 'Trained model not found! Please register a student and click "Save Profile" first.')
        return

    # Load persistent student records
    records = load_student_records(DEFAULT_STUDENT_CSV)
    if not records:
        mess.showwarning('Details Missing', 'No registered students found. Please register students first!')
        return

    student_mapping = get_student_mapping(records)

    # Initialize LBPH Recognizer
    try:
        if hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            recognizer = cv2.face_LBPHFaceRecognizer.create()
        recognizer.read(DEFAULT_MODEL_FILE)
    except Exception as e:
        mess.showerror('Recognizer Error', f"Failed to load trained model: {e}")
        return

    face_cascade = cv2.CascadeClassifier(DEFAULT_CASCADE_PATH)

    # Attendance file for today
    today_attendance_file = get_attendance_filepath(DEFAULT_ATTENDANCE_DIR)
    # Memory cache of normalized rolls already marked today for instant lookup
    today_marked_rolls = {r["normalized_roll"] for r in load_attendance_records(today_attendance_file)}

    # Open webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        mess.showerror('Camera Error', 'Could not open camera. Please check webcam connection.')
        return

    # Refresh table on GUI
    load_attendance_ui()

    try:
        while True:
            ret, im = cam.read()
            if not ret or im is None:
                continue

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                try:
                    predicted_serial, distance = recognizer.predict(gray[y:y + h, x:x + w])
                except Exception:
                    predicted_serial, distance = -1, 999.0

                # Identify face using documented LBPH confidence threshold
                name, roll, is_known = identify_face(
                    predicted_serial,
                    distance,
                    student_mapping,
                    threshold=LBPH_CONFIDENCE_THRESHOLD
                )

                status_msg = ""
                if is_known:
                    norm_roll = normalize_roll_number(roll)
                    # Check if attendance is already recorded today
                    if norm_roll in today_marked_rolls:
                        status_msg = "Attendance already marked today"
                    else:
                        # Record attendance according to workflow
                        marked, mark_msg = mark_attendance(
                            today_attendance_file,
                            roll,
                            name,
                            cached_set=today_marked_rolls
                        )
                        if marked:
                            status_msg = "Attendance Marked"
                            # Immediately update the GUI table
                            load_attendance_ui()
                        else:
                            status_msg = "Attendance already marked today"

                # Always draw identity above the face box on the live frame
                draw_face_annotations(im, x, y, w, h, name, roll, is_known, status_msg)

            # Top instructions overlay
            cv2.putText(
                im,
                "Taking Attendance | Press 'q' to stop",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.imshow('Taking Attendance', im)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Handle OpenCV window close 'X' button
            try:
                if cv2.getWindowProperty('Taking Attendance', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

    finally:
        cam.release()
        cv2.destroyAllWindows()
        # Final refresh of attendance table
        load_attendance_ui()


############################################# UI DATA LOADERS ################################################

def load_registered_students_ui():
    """
    Reads authoritative student records from StudentDetails.csv
    and refreshes the Registered Students Treeview and count label.
    Shows empty-state message if no students exist.
    """
    records = load_student_records(DEFAULT_STUDENT_CSV)
    
    # Clear existing items in Treeview
    for item in tv_students.get_children():
        tv_students.delete(item)

    total_count = len(records)
    lbl_registered_count.configure(text=f"Total Registered Students: {total_count}")
    message.configure(text=f"Total Registrations till now : {total_count}")

    if total_count == 0:
        # Show clean empty state row
        tv_students.insert('', 'end', text="--", values=("No registered students found", "Register in 'Register' tab"))
    else:
        for r in records:
            # Display: ID (serial) in #0, Roll Number, Student Name
            tv_students.insert('', 'end', text=str(r["serial"]), values=(r["roll"], r["name"]))


def load_attendance_ui():
    """Loads today's attendance records into the attendance Treeview tv."""
    for k in tv.get_children():
        tv.delete(k)

    today_file = get_attendance_filepath(DEFAULT_ATTENDANCE_DIR)
    records = load_attendance_records(today_file)
    for r in records:
        tv.insert('', 'end', text=str(r["roll"]), values=(r["name"], r["date"], r["time"]))


def show_registered_tab():
    """Switches the notebook to the Registered Students tab."""
    nb_reg.select(tab_students)
    load_registered_students_ui()


def audit_duplicates_dialog():
    """Audits the student CSV file and reports any duplicate roll numbers found."""
    report = audit_and_report_duplicates(DEFAULT_STUDENT_CSV)
    if not report["has_duplicates"]:
        mess.showinfo(
            "Data Integrity Check",
            f"All {report['total_records']} registered student roll numbers are unique and healthy!"
        )
    else:
        dup_details = []
        for norm_roll, recs in report["duplicates"].items():
            names = ", ".join([f"{r['name']} (ID {r['serial']})" for r in recs])
            dup_details.append(f"- Roll: {norm_roll} -> {names}")
        details_str = "\n".join(dup_details)
        mess.showwarning(
            "Duplicate Roll Numbers Detected",
            f"Found duplicate records in StudentDetails.csv:\n\n{details_str}\n\n"
            "Please review StudentDetails.csv to reconcile redundant entries."
        )


######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, True)
window.title("Attendance System")
window.configure(background='#262523')

# Apply modern styling
style = ttk.Style()
style.theme_use('clam')
style.configure("TNotebook", background="#00aeff", borderwidth=0)
style.configure("TNotebook.Tab", font=('times', 12, 'bold'), padding=[12, 5])
style.map("TNotebook.Tab", background=[("selected", "#3ece48"), ("!selected", "#c4c6ce")])

style.configure("Treeview", font=('times', 11), rowheight=24)
style.configure("Treeview.Heading", font=('times', 12, 'bold'))

# Frame 1: Attendance Section
frame1 = tk.Frame(window, bg="#00aeff", bd=2, relief='groove')
frame1.place(relx=0.04, rely=0.17, relwidth=0.45, relheight=0.80)

# Frame 2: Registration & Student List Container
frame2 = tk.Frame(window, bg="#00aeff", bd=2, relief='groove')
frame2.place(relx=0.51, rely=0.17, relwidth=0.45, relheight=0.80)

# Main Title Header
message3 = tk.Label(
    window,
    text="Face Recognition Based Attendance System",
    fg="white",
    bg="#262523",
    font=('times', 27, 'bold')
)
message3.place(relx=0.5, y=30, anchor='center')

# Date & Clock frames
ts = time.time()
date_obj = datetime.datetime.fromtimestamp(ts)
day = date_obj.strftime('%d')
month_name = date_obj.strftime('%B')
year = date_obj.strftime('%Y')

frame4 = tk.Frame(window, bg="#262523")
frame4.place(relx=0.32, rely=0.09, relwidth=0.20, relheight=0.06)
datef = tk.Label(
    frame4,
    text=f"{day}-{month_name}-{year}  |",
    fg="orange",
    bg="#262523",
    font=('times', 18, 'bold')
)
datef.pack(fill='both', expand=1)

frame3 = tk.Frame(window, bg="#262523")
frame3.place(relx=0.52, rely=0.09, relwidth=0.15, relheight=0.06)
clock = tk.Label(frame3, fg="orange", bg="#262523", font=('times', 18, 'bold'))
clock.pack(fill='both', expand=1)
tick()

# ----------------- FRAME 1 (ATTENDANCE) -----------------
head1 = tk.Label(
    frame1,
    text="For Already Registered",
    fg="black",
    bg="#3ece48",
    font=('times', 16, 'bold'),
    height=1
)
head1.pack(fill='x', padx=0, pady=0)

trackImg = tk.Button(
    frame1,
    text="Take Attendance",
    command=TrackImages,
    fg="black",
    bg="yellow",
    width=25,
    height=1,
    activebackground="white",
    font=('times', 14, 'bold')
)
trackImg.pack(pady=(15, 5))

lbl3 = tk.Label(
    frame1,
    text="Today's Attendance",
    fg="black",
    bg="#00aeff",
    font=('times', 15, 'bold')
)
lbl3.pack(pady=(5, 5))

# Attendance Treeview
tree_frame = tk.Frame(frame1, bg="#00aeff")
tree_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

tv = ttk.Treeview(tree_frame, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=120, anchor='center')
tv.column('name', width=150, anchor='w')
tv.column('date', width=110, anchor='center')
tv.column('time', width=100, anchor='center')

tv.heading('#0', text='ROLL NUMBER')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=tv.yview)
tv.configure(yscrollcommand=scroll.set)
scroll.pack(side='right', fill='y')
tv.pack(side='left', fill='both', expand=True)

quitWindow = tk.Button(
    frame1,
    text="Quit",
    command=window.destroy,
    fg="black",
    bg="#ea2a2a",
    width=25,
    height=1,
    activebackground="white",
    font=('times', 13, 'bold')
)
quitWindow.pack(pady=(0, 10))

# ----------------- FRAME 2 (NOTEBOOK: REGISTRATION & REGISTERED STUDENTS) -----------------
nb_reg = ttk.Notebook(frame2)
nb_reg.pack(fill='both', expand=True)

# TAB 1: Register Student
tab_register = tk.Frame(nb_reg, bg="#00aeff")
nb_reg.add(tab_register, text="  Register Student  ")

head2 = tk.Label(
    tab_register,
    text="For New Registrations",
    fg="black",
    bg="#3ece48",
    font=('times', 16, 'bold'),
    height=1
)
head2.pack(fill='x', padx=0, pady=0)

form_frame = tk.Frame(tab_register, bg="#00aeff")
form_frame.pack(fill='x', padx=20, pady=10)

# Roll Number Field
lbl_roll = tk.Label(form_frame, text="Enter Roll Number / ID", fg="black", bg="#00aeff", font=('times', 14, 'bold'))
lbl_roll.grid(row=0, column=0, sticky='w', pady=(5, 2))

txt_frame = tk.Frame(form_frame, bg="#00aeff")
txt_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
txt = tk.Entry(txt_frame, width=28, fg="black", font=('times', 14, 'bold'))
txt.pack(side='left', fill='x', expand=True)
clearButton = tk.Button(txt_frame, text="Clear", command=clear, fg="white", bg="#ea2a2a", width=7, activebackground="white", font=('times', 10, 'bold'))
clearButton.pack(side='left', padx=(5, 0))

# Student Name Field
lbl_name = tk.Label(form_frame, text="Enter Student Name", fg="black", bg="#00aeff", font=('times', 14, 'bold'))
lbl_name.grid(row=2, column=0, sticky='w', pady=(5, 2))

txt2_frame = tk.Frame(form_frame, bg="#00aeff")
txt2_frame.grid(row=3, column=0, sticky='ew', pady=(0, 10))
txt2 = tk.Entry(txt2_frame, width=28, fg="black", font=('times', 14, 'bold'))
txt2.pack(side='left', fill='x', expand=True)
clearButton2 = tk.Button(txt2_frame, text="Clear", command=clear2, fg="white", bg="#ea2a2a", width=7, activebackground="white", font=('times', 10, 'bold'))
clearButton2.pack(side='left', padx=(5, 0))

message1 = tk.Label(
    tab_register,
    text="1) Take Images  >>>  2) Save Profile",
    bg="#00aeff",
    fg="black",
    font=('times', 13, 'bold'),
    wraplength=450
)
message1.pack(pady=5)

takeImg = tk.Button(
    tab_register,
    text="1. Take Images",
    command=TakeImages,
    fg="white",
    bg="blue",
    width=28,
    height=1,
    activebackground="white",
    font=('times', 13, 'bold')
)
takeImg.pack(pady=5)

trainImg = tk.Button(
    tab_register,
    text="2. Save Profile (Train Model)",
    command=psw,
    fg="white",
    bg="blue",
    width=28,
    height=1,
    activebackground="white",
    font=('times', 13, 'bold')
)
trainImg.pack(pady=5)

btn_view_students = tk.Button(
    tab_register,
    text="View Registered Students List",
    command=show_registered_tab,
    fg="black",
    bg="#3ece48",
    width=28,
    height=1,
    activebackground="white",
    font=('times', 12, 'bold')
)
btn_view_students.pack(pady=10)

message = tk.Label(
    tab_register,
    text="Total Registrations till now : 0",
    bg="#00aeff",
    fg="black",
    font=('times', 13, 'bold')
)
message.pack(pady=5)

# TAB 2: Registered Students
tab_students = tk.Frame(nb_reg, bg="#00aeff")
nb_reg.add(tab_students, text="  Registered Students  ")

head_students = tk.Label(
    tab_students,
    text="Registered Students Directory",
    fg="black",
    bg="#3ece48",
    font=('times', 16, 'bold'),
    height=1
)
head_students.pack(fill='x', padx=0, pady=0)

top_students_bar = tk.Frame(tab_students, bg="#00aeff")
top_students_bar.pack(fill='x', padx=10, pady=5)

lbl_registered_count = tk.Label(
    top_students_bar,
    text="Total Registered Students: 0",
    fg="black",
    bg="#00aeff",
    font=('times', 13, 'bold')
)
lbl_registered_count.pack(side='left', padx=5)

btn_refresh_students = tk.Button(
    top_students_bar,
    text="Refresh List",
    command=load_registered_students_ui,
    fg="black",
    bg="yellow",
    font=('times', 11, 'bold')
)
btn_refresh_students.pack(side='right', padx=5)

# Registered Students Treeview
tree_students_frame = tk.Frame(tab_students, bg="#00aeff")
tree_students_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

tv_students = ttk.Treeview(tree_students_frame, height=13, columns=('roll', 'name'))
tv_students.column('#0', width=80, anchor='center')
tv_students.column('roll', width=160, anchor='center')
tv_students.column('name', width=200, anchor='w')

tv_students.heading('#0', text='STUDENT ID')
tv_students.heading('roll', text='ROLL NUMBER')
tv_students.heading('name', text='STUDENT NAME')

scroll_students = ttk.Scrollbar(tree_students_frame, orient='vertical', command=tv_students.yview)
tv_students.configure(yscrollcommand=scroll_students.set)
scroll_students.pack(side='right', fill='y')
tv_students.pack(side='left', fill='both', expand=True)

# Tab change event to refresh student list
def on_tab_changed(event):
    selected_tab = event.widget.tab('current')['text']
    if "Registered Students" in selected_tab:
        load_registered_students_ui()

nb_reg.bind("<<NotebookTabChanged>>", on_tab_changed)

##################### MENUBAR #################################

menubar = tk.Menu(window, relief='ridge')

filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', font=('times', 14, 'bold'), menu=filemenu)

students_menu = tk.Menu(menubar, tearoff=0)
students_menu.add_command(label='View Registered Students', command=show_registered_tab)
students_menu.add_command(label='Refresh Students List', command=load_registered_students_ui)
students_menu.add_command(label='Check Duplicate Records', command=audit_duplicates_dialog)
menubar.add_cascade(label='Students', font=('times', 14, 'bold'), menu=students_menu)

window.configure(menu=menubar)

##################### INITIAL LOAD ############################

# Populate tables and count on startup
load_registered_students_ui()
load_attendance_ui()

# Audit for existing duplicates on startup (report without altering data)
audit_info = audit_and_report_duplicates(DEFAULT_STUDENT_CSV)
if audit_info["has_duplicates"]:
    print(f"Warning: Existing duplicate roll numbers found in {DEFAULT_STUDENT_CSV}: {list(audit_info['duplicates'].keys())}")

##################### MAINLOOP ################################

if __name__ == "__main__":
    window.mainloop()
