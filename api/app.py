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

# ─── อิมพอร์ตเพิ่มเติมสำหรับระบบเสียง (Wav2Vec) ───
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Visual Models (C:)
FINE_TUNED_MODEL = os.path.join(MODEL_DIR, "finetuned_xception_model.h5")
BASE_MODEL = os.path.join(MODEL_DIR, "best_xception_model.h5")

# Audio Model (D:) - ข้ามไปดึงที่ฮาร์ดดิสก์อีกลูก
WAV2VEC_MODEL_DIR = r"D:\Wan2Training\audio_results_final_attempt\checkpoint-1787"

IMG_SIZE = (299, 299)
FACE_MARGIN = 0.05   # 5% margin around detected face
VIDEO_FRAME_COUNT = 5
FAKE_THRESHOLD = 0.5

# ─── App & CORS ───────────────────────────────────────────────────────────────

app = FastAPI(title="Monolith Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading (ภาพและเสียง) ──────────────────────────────────────────────

def _load_visual_model():
    if os.path.exists(FINE_TUNED_MODEL):
        print(f"🎯 โหลดโมเดลภาพ (Fine-tune): {FINE_TUNED_MODEL}")
        return tf.keras.models.load_model(FINE_TUNED_MODEL)
    if os.path.exists(BASE_MODEL):
        print(f"📦 โหลดโมเดลภาพ (หลัก): {BASE_MODEL}")
        return tf.keras.models.load_model(BASE_MODEL)
    print("⚠️  ไม่พบไฟล์โมเดลภาพในโฟลเดอร์ models")
    return None

try:
    model = _load_visual_model()
except Exception as e:
    print(f"❌ โหลดโมเดลภาพไม่สำเร็จ: {e}")
    model = None

# ─── โซนโหลดโมเดลเสียง Wav2Vec (Offline Mode) ───
try:
    import traceback
    print(f"🎧 กำลังโหลดระบบการฟัง (Wav2Vec) จาก: {WAV2VEC_MODEL_DIR}")
    
    # 🌟 ดึงข้อมูล Extractor จากไฟล์ preprocessor_config.json ที่คุณเพิ่งก๊อปไปวาง
    audio_extractor = AutoFeatureExtractor.from_pretrained(WAV2VEC_MODEL_DIR)
    
    # 🌟 ดึงสมอง (Weights) จากไฟล์ safetensors
    audio_model = AutoModelForAudioClassification.from_pretrained(WAV2VEC_MODEL_DIR, use_safetensors=True)
    
    print("✅ โหลดโมเดลเสียงสำเร็จ! ระบบ Monolith พร้อมทำงาน 100%")
except Exception as e:
    print("\n" + "="*50)
    print("🚨 แฉต้นตอ ERROR (ก๊อปตรงนี้มาให้ผมดูเลยครับ):")
    traceback.print_exc()
    print("="*50 + "\n")
    print(f"❌ โหลดโมเดลเสียงไม่สำเร็จ: {e}")
    audio_model = None
    audio_extractor = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─── Face Utilities ───────────────────────────────────────────────────────────

def _expand_bbox(x, y, w, h, iw, ih, margin: float = FACE_MARGIN):
    mx = int(w * margin)
    my = int(h * margin)
    return (
        max(0, x - mx),
        max(0, y - my),
        min(iw, x + w + mx),
        min(ih, y + h + my),
    )

def detect_faces(gray_image):
    return face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

def crop_face(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return image_array
    x, y, w, h = faces[0]
    ih, iw = image_array.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(x, y, w, h, iw, ih)
    return image_array[y1:y2, x1:x2]

# ─── Preprocessing ────────────────────────────────────────────────────────────

def prepare_image(image_pil: Image.Image) -> np.ndarray:
    img_array = np.array(image_pil.convert("RGB"))
    face_array = crop_face(img_array)
    resized = Image.fromarray(face_array).resize(IMG_SIZE)
    normalised = np.array(resized) / 255.0
    return np.expand_dims(normalised, axis=0)

# ─── Prediction Helpers (Visual) ──────────────────────────────────────────────

def _score_to_verdict(score: float) -> tuple[str, float]:
    is_fake = score < FAKE_THRESHOLD
    label = "FAKE" if is_fake else "REAL"
    conf = round((1 - score if is_fake else score) * 100, 2)
    return label, conf

def _draw_face_box(bgr_image: np.ndarray, label: str, conf: float) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return bgr_image
    x, y, w, h = faces[0]
    color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), color, 3)
    cv2.putText(
        bgr_image,
        f"{label} {conf}%",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return bgr_image

def _encode_frame_base64(bgr_frame: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", bgr_frame)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def _predict_face_in_frame(bgr_frame: np.ndarray) -> float | None:
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    ih, iw = bgr_frame.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(x, y, w, h, iw, ih)
    face_bgr = bgr_frame[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    processed = prepare_image(Image.fromarray(face_rgb))
    return float(model.predict(processed, verbose=0)[0][0])

# ─── Audio Processing & Prediction ────────────────────────────────────────────

def predict_audio_file(audio_path: str):
    """สกัดเสียงและวิเคราะห์ผ่าน Wav2Vec"""
    if audio_model is None or audio_extractor is None:
        raise RuntimeError("โมเดล Wav2Vec ยังไม่พร้อมทำงาน")

    # Wav2Vec2 ต้องใช้ Sample Rate 16000 Hz เสมอ
    speech, rate = librosa.load(audio_path, sr=16000)
    
    # แปลงคลื่นเสียงเป็น Tensors ให้ Pytorch เข้าใจ
    inputs = audio_extractor(speech, sampling_rate=rate, return_tensors="pt")
    
    # ส่งเข้าโมเดลทำนายผล
    with torch.no_grad():
        logits = audio_model(**inputs).logits
        scores = F.softmax(logits, dim=1).numpy()[0]
        
    # ⚠️ หมายเหตุ: ตรงนี้ถือว่า Class 1 = FAKE และ Class 0 = REAL 
    # (ถ้าเพื่อนเทรนมาสลับกัน ให้เปลี่ยนเป็น `scores[0] > scores[1]`)
    is_fake = scores[1] > scores[0] 
    
    label = "FAKE" if is_fake else "REAL"
    conf = float(max(scores) * 100)
    
    return label, round(conf, 2)

# ─── Video Processing ─────────────────────────────────────────────────────────

def predict_video_frames(video_path: str, frame_count: int = VIDEO_FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV ไม่สามารถเปิดไฟล์วิดีโอได้")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("วิดีโอไม่มีเฟรมภาพให้ตรวจ")

    scores = []
    annotated_frames = []
    indices = np.linspace(0, total_frames - 1, min(frame_count, total_frames), dtype=int)

    for frame_id in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        score = _predict_face_in_frame(frame)
        if score is not None:
            scores.append(score)
            label, conf = _score_to_verdict(score)
            frame = _draw_face_box(frame, label, conf)

        annotated_frames.append(_encode_frame_base64(frame))

    cap.release()
    avg_score = float(np.mean(scores)) if scores else FAKE_THRESHOLD
    return avg_score, annotated_frames

def extract_audio_from_video(video_path: str) -> str | None:
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

# ─── Model Guards ─────────────────────────────────────────────────────────────

def _require_visual_model():
    if model is None:
        raise HTTPException(status_code=503, detail="โมเดลภาพยังไม่พร้อมทำงาน")

def _require_audio_model():
    if audio_model is None:
        raise HTTPException(status_code=503, detail="โมเดลเสียง (Wav2Vec) ยังไม่พร้อมทำงาน")

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "status": "online",
        "owner": "Mo",
        "project": "Monolith",
        "visual_model_loaded": model is not None,
        "audio_model_loaded": audio_model is not None,
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    _require_visual_model()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดรูปภาพ")

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("ไฟล์ภาพว่างเปล่า")

        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        processed = prepare_image(img_pil)

        score = float(model.predict(processed, verbose=0)[0][0])
        label, conf = _score_to_verdict(score)

        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_bgr = _draw_face_box(img_bgr, label, conf)
        annotated_frame = _encode_frame_base64(img_bgr)

        return {
            "success": True,
            "label": label,
            "confidence": conf,
            "frames": [annotated_frame],
            "message": "AI วิเคราะห์โครงสร้างใบหน้าสำเร็จ",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Image Error: {e}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {e}")

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """API ใหม่: รับไฟล์เสียง WAV เข้ามาวิเคราะห์ด้วย Wav2Vec"""
    _require_audio_model()

    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์เสียง")

    temp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("ไฟล์เสียงว่างเปล่า")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        # วิเคราะห์ผลผ่านฟังก์ชัน
        label, conf = predict_audio_file(temp_path)

        return {
            "success": True,
            "label": label,
            "confidence": conf,
            "message": "วิเคราะห์คลื่นเสียงสำเร็จด้วย Wav2Vec 2.0",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Audio Error: {e}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการวิเคราะห์เสียง: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    _require_visual_model()

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดวิดีโอ")

    temp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("ไฟล์วิดีโอว่างเปล่า")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        avg_score, frames = predict_video_frames(temp_path)
        label, conf = _score_to_verdict(avg_score)
        
        audio_path = extract_audio_from_video(temp_path)
        
        # ถ้าระบบเสียงพร้อม และแยกไฟล์เสียงออกมาได้ ให้โยนเข้า Wav2Vec ต่อเลย!
        audio_label = None
        audio_conf = None
        if audio_model is not None and audio_path is not None:
            try:
                audio_label, audio_conf = predict_audio_file(audio_path)
            except Exception as e:
                print(f"⚠️ สแกนเสียงในวิดีโอพลาด: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        return {
            "success": True,
            "visual_prediction": label,
            "visual_confidence": conf,
            "audio_prediction": audio_label,    # ส่งผลลัพธ์ของเสียงแนบไปด้วย
            "audio_confidence": audio_conf,     # คืนค่าความมั่นใจของฝั่งเสียง
            "frames": frames,
            "message": "วิเคราะห์เฟรมวิดีโอและเสียงเสร็จสมบูรณ์",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Video Error: {e}")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการวิเคราะห์วิดีโอ: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)