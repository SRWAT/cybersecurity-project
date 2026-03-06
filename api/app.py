from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
import os
import cv2
import tempfile
import moviepy as mp
import base64
from PIL import Image

# --- 1. ตั้งค่าระบบ API และ CORS ---
app = FastAPI(title="Monolith Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. การจัดการตำแหน่งไฟล์โมเดล ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

FINE_TUNED_MODEL = os.path.join(MODEL_DIR, "finetuned_xception_model.h5")
BASE_MODEL = os.path.join(MODEL_DIR, "best_xception_model.h5")

# --- 3. โหลดโมเดล AI ---
try:
    if os.path.exists(FINE_TUNED_MODEL):
        print(f"🎯 Mo! โหลดโมเดล Fine-tune: {FINE_TUNED_MODEL}")
        model = tf.keras.models.load_model(FINE_TUNED_MODEL)
    elif os.path.exists(BASE_MODEL):
        print(f"📦 โหลดโมเดลหลัก (80%): {BASE_MODEL}")
        model = tf.keras.models.load_model(BASE_MODEL)
    else:
        model = None
        print("⚠️ คำเตือน: ยังไม่มีไฟล์โมเดลในโฟลเดอร์ models")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    model = None

# --- 4. ฟังก์ชันช่วยเหลือ (Utility Functions) ---

# ใช้ระบบสแกนใบหน้าของ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_only(image_pil):
    """ใช้ OpenCV สแกนหาใบหน้าและตัดมาเฉพาะหน้า (บีบขอบ 5% ลด Noise)"""
    img_rgb = image_pil.convert('RGB')
    img_array = np.array(img_rgb)
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        ih, iw, _ = img_array.shape
        
        # เผื่อขอบแค่ 5% ให้ AI โฟกัสแค่หน้า
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(iw, x + w + margin_x)
        y2 = min(ih, y + h + margin_y)

        cropped_face = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped_face)
    
    return image_pil

def prepare_image(image_pil):
    """เตรียมรูปภาพ: ครอปหน้า -> ย่อขนาด 299x299 -> Normalization"""
    face_image = get_face_only(image_pil)
    img = face_image.resize((299, 299))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_audio_from_video(video_path):
    """สกัดเสียงจากวิดีโอออกมาเป็นไฟล์ .wav"""
    try:
        audio_output = video_path.replace(".mp4", ".wav")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output, logger=None)
        video.audio.close()
        video.close()
        return audio_output
    except Exception as e:
        print(f"❌ สกัดเสียงไม่สำเร็จ: {e}")
        return None

# ก๊อปปี้ไปวางทับฟังก์ชัน predict_video_frames เดิมใน app.py
def predict_video_frames(video_path, frame_count=5):
    """ดึงเฟรมแบบกระจายช่วงเวลาให้คงที่ วาดกรอบใบหน้า และแปลงเป็นรูปภาพส่งให้หน้าเว็บ"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ OpenCV ไม่สามารถเปิดไฟล์วิดีโอได้")
        return 0.5, []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("❌ วิดีโอไม่มีเฟรมภาพให้ตรวจ")
        return 0.5, []
        
    scores = []
    annotated_frames = []

    # 🔥 จุดที่แก้: ใช้ np.linspace แบ่งช่วงวิดีโอให้เท่าๆ กัน แทนการสุ่ม (Random)
    actual_frame_count = min(frame_count, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, actual_frame_count, dtype=int)

    for frame_id in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        score_val = 0.5
        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)

            cropped_face = frame[y1:y2, x1:x2]
            img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            processed_img = prepare_image(img_pil)
            prediction = model.predict(processed_img, verbose=0)
            score_val = float(prediction[0][0])
            scores.append(score_val)

            # วาดกรอบสี่เหลี่ยม
            is_fake = score_val < 0.5
            label = "FAKE" if is_fake else "REAL"
            conf = round((1-score_val if is_fake else score_val) * 100, 2)
            color = (0, 0, 255) if is_fake else (0, 255, 0) # BGR

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"{label} {conf}%", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # แปลงรูปที่วาดกรอบแล้วเป็นข้อความ (Base64)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_frames.append(f"data:image/jpeg;base64,{frame_base64}")
            
    cap.release()
    avg_score = float(np.mean(scores)) if scores else 0.5
    return avg_score, annotated_frames

# --- 5. Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "owner": "Mo", "project": "Monolith"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดรูปภาพ")
    
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("ไฟล์ภาพว่างเปล่า (Empty File)")
            
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        processed_img = prepare_image(img_pil)
        
        prediction = model.predict(processed_img, verbose=0)
        score = float(prediction[0][0])
        is_fake = score < 0.5
        
        # --- เพิ่มการวาดกรอบสำหรับรูปภาพ ---
        label = "FAKE" if is_fake else "REAL"
        conf = round((1-score if is_fake else score) * 100, 2)
        
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            color = (0, 0, 255) if is_fake else (0, 255, 0)
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_cv, f"{label} {conf}%", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        _, buffer = cv2.imencode('.jpg', img_cv)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_frames = [f"data:image/jpeg;base64,{frame_base64}"]
        # ------------------------------------

        return {
            "success": True,
            "label": label,
            "confidence": conf,
            "frames": annotated_frames,
            "message": "AI วิเคราะห์เฉพาะบริเวณโครงสร้างใบหน้า (OpenCV)"
        }
    except Exception as e:
        print(f"❌ Image Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดวิดีโอ")

    temp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("ไฟล์วิดีโอว่างเปล่า (Empty File)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_v:
            temp_v.write(contents)
            temp_path = temp_v.name

        avg_score, frames = predict_video_frames(temp_path)
        is_fake_video = avg_score < 0.5
        audio_path = extract_audio_from_video(temp_path)

        return {
            "success": True,
            "visual_prediction": "FAKE" if is_fake_video else "REAL",
            "visual_confidence": round((1-avg_score if is_fake_video else avg_score) * 100, 2),
            "audio_extracted": True if audio_path else False,
            "audio_file": os.path.basename(audio_path) if audio_path else None,
            "frames": frames,
            "message": "วิเคราะห์เฟรมและสกัดเสียงสำเร็จ พร้อมส่งเข้า Wav2Vec"
        }
    except Exception as e:
        print(f"❌ Video Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการวิเคราะห์วิดีโอ: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)