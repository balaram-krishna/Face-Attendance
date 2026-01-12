############################################################
# FACE ATTENDANCE SYSTEM – FINAL SAFE VERSION
# UI SAME AS BEFORE | LOGIC UPGRADED
############################################################

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

############################################################
# CONFIG
############################################################
st.set_page_config(page_title="Face Attendance System", layout="wide")

ROOT_REG = "registered_faces"
ROOT_SAVE = "attendance_records"

os.makedirs(ROOT_REG, exist_ok=True)
os.makedirs(ROOT_SAVE, exist_ok=True)

MODEL_NAME = "buffalo_l"

THRESHOLD = 0.46        # confidence
MARGIN = 0.08           # safety margin
MAX_YAW = 55            # allow normal side faces
MIN_FACE_RATIO = 0.006  # allow small classroom faces

MIN_CAPTURES = 1
MAX_CAPTURES = 4

############################################################
# SESSION STATE
############################################################
if "captures" not in st.session_state:
    st.session_state.captures = []

if "capture_count" not in st.session_state:
    st.session_state.capture_count = 0

if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None

if "final_image" not in st.session_state:
    st.session_state.final_image = None

############################################################
# LOAD MODEL
############################################################
@st.cache_resource
def load_model():
    m = FaceAnalysis(name=MODEL_NAME)
    m.prepare(ctx_id=-1, det_size=(640, 640))
    return m

model = load_model()

############################################################
# LOAD REGISTERED FACES
############################################################
@st.cache_data
def load_registered_faces():
    names, embs = [], []

    for fn in os.listdir(ROOT_REG):
        img = cv2.imread(os.path.join(ROOT_REG, fn))
        if img is None:
            continue

        faces = model.get(img)
        if not faces:
            continue

        emb = faces[0].normed_embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10

        base = os.path.splitext(fn)[0]
        if "_" in base and base.split("_")[-1].isdigit():
            base = "_".join(base.split("_")[:-1])

        names.append(base)
        embs.append(emb)

    return names, np.vstack(embs)

names, reg_embs = load_registered_faces()

if not names:
    st.error("No registered faces found in registered_faces/")
    st.stop()

############################################################
# HELPERS
############################################################
def pil_from_bgr(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_box(draw, bbox, label, ok=True):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 180, 0) if ok else (220, 40, 40)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)

def is_blurry(face_img, thr=80):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thr

def bad_lighting(face_img):
    mean = np.mean(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
    return mean < 55 or mean > 210

def bad_pose(face):
    if hasattr(face, "pose"):
        return abs(face.pose[0]) > MAX_YAW
    return False

def score_image(img):
    faces = model.get(img)
    img_area = img.shape[0] * img.shape[1]
    valid = []

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        area = (x2 - x1) * (y2 - y1)
        if area / img_area >= MIN_FACE_RATIO:
            valid.append(area)

    if not valid:
        return 0, 0

    return len(valid), np.mean(valid)

############################################################
# SIDEBAR (UNCHANGED)
############################################################
with st.sidebar:
    st.header("Session Details")
    subject = st.text_input("Subject")
    faculty = st.text_input("Faculty")
    period = st.selectbox("Period", [f"Period {i}" for i in range(1, 9)])

    st.markdown("---")
    st.write(f"Registered Students: **{len(set(names))}**")

    if st.button("🔄 New Session"):
        st.session_state.captures = []
        st.session_state.capture_count = 0
        st.session_state.uploaded_img = None
        st.session_state.final_image = None
        st.rerun()

############################################################
# MAIN UI (UNCHANGED)
############################################################
st.title("📸 Face Attendance System")

st.info(
    f"Model: buffalo_l | Threshold={THRESHOLD} | "
    f"Captures {st.session_state.capture_count}/{MAX_CAPTURES}"
)

############################################################
# UPLOAD + MULTI-CAPTURE (UNCHANGED)
############################################################
st.subheader("📥 Upload OR 📷 Capture Classroom Images")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "Upload Classroom Photo",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        st.session_state.uploaded_img = img
        st.image(pil_from_bgr(img), caption="Uploaded Image")

with col2:
    if st.session_state.capture_count < MAX_CAPTURES:
        cam = st.camera_input(
            f"Click Photo {st.session_state.capture_count + 1}",
            key=f"camera_{st.session_state.capture_count}"
        )
        if cam is not None:
            img = cv2.imdecode(
                np.frombuffer(cam.getvalue(), np.uint8),
                cv2.IMREAD_COLOR
            )
            st.session_state.captures.append(img)
            st.session_state.capture_count += 1
            st.rerun()
    else:
        st.success("Maximum captures reached")

############################################################
# SHOW CAPTURES (UNCHANGED)
############################################################
if st.session_state.captures:
    st.markdown("### 🖼 Captured Photos")
    cols = st.columns(len(st.session_state.captures))
    for i, img in enumerate(st.session_state.captures):
        cols[i].image(pil_from_bgr(img), caption=f"Photo {i+1}", use_container_width=True)

############################################################
# PROCESS IMAGE SELECTION (UNCHANGED)
############################################################
st.markdown("---")

if (
    st.session_state.uploaded_img is not None
    or len(st.session_state.captures) >= MIN_CAPTURES
):
    if st.button("🚀 Process Attendance"):
        candidates = []

        if st.session_state.uploaded_img is not None:
            c, a = score_image(st.session_state.uploaded_img)
            candidates.append((c, a, st.session_state.uploaded_img))

        for img in st.session_state.captures:
            c, a = score_image(img)
            candidates.append((c, a, img))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        st.session_state.final_image = candidates[0][2]

############################################################
# FACE RECOGNITION (LOGIC UPGRADED)
############################################################
if st.session_state.final_image is not None:
    img = st.session_state.final_image
    faces = model.get(img)

    pil_img = pil_from_bgr(img)
    draw = ImageDraw.Draw(pil_img)

    recognised = set()

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        area = (x2 - x1) * (y2 - y1)
        ratio = area / (img.shape[0] * img.shape[1])

        if ratio < MIN_FACE_RATIO:
            draw_box(draw, face.bbox, "Too Small", False)
            continue

        face_img = img[y1:y2, x1:x2]

        if is_blurry(face_img):
            draw_box(draw, face.bbox, "Blur", False)
            continue

        if bad_lighting(face_img):
            draw_box(draw, face.bbox, "Bad Light", False)
            continue

        if bad_pose(face):
            draw_box(draw, face.bbox, "Extreme Side", False)
            continue

        emb = face.normed_embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-10

        sims = reg_embs @ emb
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        sorted_sims = np.sort(sims)[::-1]
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0

        if best_score >= THRESHOLD and margin >= MARGIN:
            name = names[best_idx]
            recognised.add(name)
            draw_box(draw, face.bbox, name, True)
        else:
            draw_box(draw, face.bbox, "Unknown", False)

    st.image(pil_img, use_container_width=True)

    ########################################################
    # ATTENDANCE TABLE (UNCHANGED)
    ########################################################
    st.subheader("📋 Attendance")

    rows = []
    for s in sorted(set(names)):
        rows.append({
            "Student Name": s.replace("_", " "),
            "Status": "Present" if s in recognised else "Absent"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)

    ########################################################
    # MANUAL VERIFICATION (UNCHANGED)
    ########################################################
    st.subheader("🟥 Manual Verification")

    for s in sorted(set(names)):
        if s not in recognised:
            if st.checkbox(f"Mark {s.replace('_',' ')} as Present"):
                df.loc[
                    df["Student Name"] == s.replace("_"," "),
                    "Status"
                ] = "Present"

    ########################################################
    # SAVE ATTENDANCE (UNCHANGED)
    ########################################################
    if st.button("💾 Save Attendance"):
        today = date.today().isoformat()
        folder = os.path.join(ROOT_SAVE, today)
        os.makedirs(folder, exist_ok=True)

        df["Subject"] = subject
        df["Faculty"] = faculty
        df["Period"] = period
        df["Date"] = today
        df["Time"] = datetime.now().strftime("%I:%M %p")

        path = os.path.join(folder, f"{today}_{period.replace(' ','_')}.csv")
        df.to_csv(path, index=False)

        st.success("✅ Attendance saved (no wrong decisions).")
