import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import random

# --- 1. โหลดโมเดลล่าสุดของคุณ Mo ---
model_path = './models/best_xception_model.h5'
model = tf.keras.models.load_model(model_path)

def test_random_images(base_path, label_name, num_tests=3):
    print(f"\n🔍 กำลังสุ่มตรวจรูปประเภท: {label_name.upper()}")
    folder_path = os.path.join(base_path, label_name)
    all_images = os.listdir(folder_path)
    sampled_images = random.sample(all_images, num_tests)

    for img_name in sampled_images:
        img_path = os.path.join(folder_path, img_name)
        
        # Pre-processing (ต้องทำเหมือนตอนเทรนเป๊ะๆ)
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0]
        
        # 0 = Fake, 1 = Real
        result = "REAL ✅" if score > 0.5 else "FAKE 🚫"
        confidence = score if score > 0.5 else (1 - score)
        
        print(f"🖼️ ไฟล์: {img_name} | AI ทายว่า: {result} (มั่นใจ {confidence*100:.2f}%)")

# --- 2. สั่งรันทดสอบด่วน ---
val_path = './data/val/'
if os.path.exists(val_path):
    test_random_images(val_path, 'fake', num_tests=3)
    test_random_images(val_path, 'real', num_tests=3)
else:
    print("❌ ไม่พบโฟลเดอร์ข้อมูลสำหรับทดสอบ!")