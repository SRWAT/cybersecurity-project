# Deepfake Detection Engine

ระบบตรวจจับ Deepfake ด้วย Deep Learning สำหรับไฟล์ภาพ วิดีโอ และเสียง  
ใช้ XceptionNet สำหรับการวิเคราะห์ภาพ และ Wav2Vec สำหรับการวิเคราะห์เสียง  
พัฒนาด้วย Nuxt 3 (Frontend) + FastAPI (Backend)

---

## Architecture

```
browser  ──►  Nuxt 3 (frontend)  ──►  FastAPI (backend)  ──►  TensorFlow model
               localhost:3000          localhost:8000
```

| Layer    | Stack                                     |
|----------|-------------------------------------------|
| Frontend | Nuxt 3, Vue 3, Tailwind CSS, WaveSurfer.js |
| Backend  | FastAPI, TensorFlow/Keras, OpenCV, MoviePy |
| Model    | XceptionNet (visual), Wav2Vec (audio)      |
| Dataset  | FaceForensics++                            |

---

## Project Structure

```
CYBERSECURITY-PROJECT/
├── app/
│   └── app.vue                    # Root Vue component + UI
│
├── server/api/
│   └── predict.post.ts            # Nuxt server route — proxy to FastAPI
│
├── src/
│   ├── prepare_data.py            # Dataset preprocessing
│   ├── train_xception.py          # Initial model training
│   ├── fine_tune.py               # Fine-tuning with resume support
│   ├── plot_history.py            # Plot training graphs
│   └── quicktest.py               # Quick inference test
│
├── models/
│   ├── best_xception_model.h5
│   ├── finetuned_xception_model.h5
│   ├── train_history.pkl
│   ├── finetune_history.pkl
│   └── training_report.png
│
├── data/                          # Dataset (ไม่ถูก track โดย git)
├── public/
├── nuxt.config.ts
├── tailwind.config.js
└── tsconfig.json
```

---

## Requirements

- Node.js 18+
- Python 3.10+
- CUDA-compatible GPU (recommended)

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd CYBERSECURITY-PROJECT
```

### 2. Node dependencies (Nuxt)

```bash
npm install
```

### 3. Python dependencies (FastAPI + ML)

```bash
pip install -r requirements.txt
```

---

## Running

### Development

Start the FastAPI backend first, then the Nuxt dev server.

```bash
# Terminal 1 — Backend (FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (Nuxt)
npm run dev
```

Frontend: `http://localhost:3000`  
Backend API: `http://localhost:8000`

### Production

```bash
# Backend
uvicorn app:app --host 0.0.0.0 --port 8000

# Frontend
npm run build
npm run preview
```

---

## Model Training

โมเดลต้องอยู่ในโฟลเดอร์ `models/` ก่อนรัน backend  
ถ้ายังไม่มีโมเดล ให้เทรนตามขั้นตอนด้านล่าง

### 1. เตรียม Dataset

```bash
python src/prepare_data.py
```

จัดโครงสร้าง dataset เป็น:

```
data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

### 2. เทรนโมเดลหลัก

```bash
python src/train_xception.py
```

บันทึก checkpoint ที่ `models/best_xception_model.h5`

### 3. Fine-tune (Optional)

```bash
python src/fine_tune.py
```

รองรับ resume — ถ้ารันซ้ำจะเทรนต่อจาก epoch ที่ค้างไว้โดยอัตโนมัติ  
บันทึก checkpoint ที่ `models/finetuned_xception_model.h5`

### 4. ดูกราฟผลการเทรน

```bash
python src/plot_history.py
```

---

## API Reference

### `POST /predict`

วิเคราะห์รูปภาพ

**Request:** `multipart/form-data` — `file` (image/*)

**Response:**
```json
{
  "success": true,
  "label": "FAKE",
  "confidence": 94.2,
  "frames": ["data:image/jpeg;base64,..."],
  "message": "..."
}
```

### `POST /predict/video`

วิเคราะห์วิดีโอ (สุ่ม 5 เฟรม + สกัดเสียง)

**Request:** `multipart/form-data` — `file` (video/mp4)

**Response:**
```json
{
  "success": true,
  "visual_prediction": "REAL",
  "visual_confidence": 87.5,
  "audio_extracted": true,
  "audio_file": "tmp_abc123.wav",
  "frames": ["data:image/jpeg;base64,..."],
  "message": "..."
}
```

### `GET /`

Health check — คืน `model_loaded: true/false`

---

## Supported Formats

| Format | Endpoint         |
|--------|------------------|
| JPG, PNG | `/predict`     |
| MP4    | `/predict/video` |
| WAV    | *(coming soon)*  |
