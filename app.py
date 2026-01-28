############################################################
# FACE ATTENDANCE SYSTEM (DB MASTER, SAFE VERSION)
############################################################

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import pyodbc

from datetime import datetime, date
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis

############################################################
# CONFIG
############################################################
st.set_page_config(page_title="Face Attendance System", layout="wide")

MODEL_NAME = "buffalo_l"
THRESHOLD = 0.46

############################################################
# DATABASE CONNECTION
############################################################
@st.cache_resource
def get_db_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=192.168.1.56,1433;"
        "DATABASE=ICFAISMS;"
        "UID=sa;"
        "PWD=icfai@123;"
        "TrustServerCertificate=yes;"
    )

db_conn = get_db_connection()

############################################################
# LOAD FACE MODEL
############################################################
@st.cache_resource
def load_model():
    model = FaceAnalysis(name=MODEL_NAME)
    model.prepare(ctx_id=-1, det_size=(640, 640))
    model.get(np.zeros((320, 320, 3), dtype=np.uint8))
    return model

model = load_model()

############################################################
# LOAD STUDENTS + REFERENCE EMBEDDINGS
############################################################
@st.cache_data
def load_students_and_embeddings(selected_class, selected_section):
    query = """
        SELECT roll_no, student_name, image_path
        FROM Students
        WHERE class = ? AND section = ?
    """
    students_df = pd.read_sql(query, db_conn, params=[selected_class, selected_section])

    known_embeddings = {}

    for _, row in students_df.iterrows():
        img_path = row["image_path"]

        if pd.isna(img_path):
            continue
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = model.get(img)
        if not faces:
            continue

        emb = faces[0].normed_embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10

        known_embeddings[row["roll_no"]] = emb

    return students_df, known_embeddings

############################################################
# HELPERS
############################################################
def pil_from_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_box(draw, bbox, label, ok=True):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 180, 0) if ok else (220, 40, 40)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1, max(0, y1 - 18)), label, fill=color)

############################################################
# SIDEBAR
############################################################
with st.sidebar:
    st.header("Session Details")
    faculty = st.text_input("Faculty Name")
    subject = st.text_input("Subject")
    selected_class = st.text_input("Class")
    selected_section = st.text_input("Section")
    period = st.selectbox("Period", [f"Period {i}" for i in range(1, 9)])

############################################################
# LOAD DATA
############################################################
students_df, known_embeddings = load_students_and_embeddings(
    selected_class, selected_section
)

st.info(f"Total students loaded from DB: {len(students_df)}")
st.info(f"Students with reference photos: {len(known_embeddings)}")

if students_df.empty:
    st.error("❌ No students found for selected class/section")
    st.stop()

############################################################
# MAIN UI
############################################################
st.title("📸 Face Attendance System")

uploaded = st.file_uploader(
    "Upload Classroom Image",
    type=["jpg", "jpeg", "png"]
)

############################################################
# ALWAYS BUILD ATTENDANCE (NO BLANK OUTPUT)
############################################################
today = date.today().isoformat()
now = datetime.now().strftime("%H:%M:%S")
recognised = set()

if uploaded:
    img = cv2.imdecode(
        np.frombuffer(uploaded.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    faces = model.get(img)
    pil_img = pil_from_bgr(img)
    draw = ImageDraw.Draw(pil_img)

    if len(faces) == 0:
        st.warning("⚠️ No faces detected in image")

    for face in faces:
        emb = face.normed_embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10

        best_roll = None
        best_score = -1

        for roll_no, ref_emb in known_embeddings.items():
            score = np.dot(ref_emb, emb)
            if score > best_score:
                best_score = score
                best_roll = roll_no

        if best_score >= THRESHOLD:
            recognised.add(best_roll)
            draw_box(draw, face.bbox, best_roll, True)
        else:
            draw_box(draw, face.bbox, "Unknown", False)

    st.image(pil_img, use_container_width=True)

else:
    st.warning("⬆️ Please upload a classroom image to process attendance")

############################################################
# ATTENDANCE TABLE (ALWAYS SHOWN)
############################################################
attendance_rows = []

for _, student in students_df.iterrows():
    rn = student["roll_no"]
    nm = student["student_name"]

    attendance_rows.append({
        "Roll No": rn,
        "Name": nm,
        "Status": "Present" if rn in recognised else "Absent"
    })

attendance_df = pd.DataFrame(attendance_rows)
st.subheader("📋 Attendance")
st.dataframe(attendance_df, hide_index=True)

############################################################
# SAVE ATTENDANCE
############################################################
if st.button("💾 Save Attendance"):
    cursor = db_conn.cursor()

    for _, r in attendance_df.iterrows():
        cursor.execute(
            """
            INSERT INTO Attendance
            (roll_no, student_name, class, section, subject,
             faculty, period, date, time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            r["Roll No"],
            r["Name"],
            selected_class,
            selected_section,
            subject,
            faculty,
            period,
            today,
            now,
            r["Status"]
        )

    db_conn.commit()
    cursor.close()
    st.success("✅ Attendance saved successfully")
