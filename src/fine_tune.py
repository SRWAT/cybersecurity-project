import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import pickle

# --- 1. การตั้งค่าสเปก ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 16 
EPOCHS = 10 # เป้าหมายสูงสุดที่เราอยากไปให้ถึง (รวมของเก่าและใหม่)
MODEL_PATH = './models/finetuned_xception_model.h5'
HISTORY_PATH = './models/finetune_history.pkl'

# --- 2. เตรียมท่อส่งข้อมูล ---
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./data/train/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_generator = val_datagen.flow_from_directory('./data/val/', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# --- 3. ระบบจำความจำ (Resume Epoch) ---
start_epoch = 0
if os.path.exists(HISTORY_PATH):
    try:
        with open(HISTORY_PATH, 'rb') as f:
            past_history = pickle.load(f)
        # ตรวจสอบว่ามีข้อมูลความแม่นยำบันทึกไว้กี่รอบแล้ว (กี่ Epoch)
        if 'accuracy' in past_history:
            start_epoch = len(past_history['accuracy'])
            print(f"🧠 AI จำได้ว่าเคยเรียนไปแล้ว {start_epoch} Epochs!")
            print(f"⏭️ จะเริ่มเทรนต่อที่ Epoch ที่ {start_epoch + 1} ไปจนถึงเป้าหมายที่ {EPOCHS}")
    except Exception as e:
        print(f"⚠️ อ่านไฟล์ประวัติไม่สำเร็จ เริ่มนับ 0 ใหม่ (Error: {e})")

# ถ้าเทรนครบเป้าหมายแล้ว ก็ไม่ต้องรันต่อ
if start_epoch >= EPOCHS:
    print(f"🎉 AI เทรนครบ {EPOCHS} Epochs ตามเป้าหมายแล้ว! ไม่จำเป็นต้องรันต่อครับ")
    exit()

# --- 4. โหลดโมเดล (เลือกว่าจะโหลดตัวใหม่สุด หรือ ตัวตั้งต้น) ---
if os.path.exists(MODEL_PATH) and start_epoch > 0:
    print(f"📦 โหลดโมเดลที่เคย Fine-tune ค้างไว้ ({MODEL_PATH}) มาทำต่อ...")
    model = load_model(MODEL_PATH)
else:
    print("📦 โหลดโมเดลหลัก (80%) ของเรามาทำการ Fine-Tune ครั้งแรก...")
    model = load_model('./models/best_xception_model.h5')

# ดึงแกนกลาง Xception ออกมาและสั่งปลดล็อก
try:
    base_model = model.layers[0] 
    base_model.trainable = True 
    for layer in base_model.layers[:100]:
        layer.trainable = False
    print(f"🔓 ปลดล็อกเลเยอร์เพื่อ Fine-tune จำนวน: {len(base_model.layers) - 100} เลเยอร์")
except Exception as e:
    print("⚠️ โมเดลนี้ถูกปลดล็อกและคอมไพล์ไปแล้ว ข้ามขั้นตอนการแช่แข็งเลเยอร์")

# --- 5. คอมไพล์ ---
# สำคัญ: แม้โหลดมาก็ควรคอมไพล์ใหม่เพื่อให้แน่ใจเรื่อง Learning Rate ต่ำๆ สำหรับการ Fine-tune
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 6. Custom Callback สำหรับเซฟประวัติ ---
class SaveHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'rb') as f:
                current_history = pickle.load(f)
            for key in logs.keys():
                if key in current_history:
                    current_history[key].append(logs[key])
                else:
                    current_history[key] = [logs[key]]
        else:
            current_history = {key: [val] for key, val in logs.items()}
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(current_history, f)

# --- 7. Callbacks มาตรฐาน ---
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- 8. เริ่มเทรนแบบ Fine-Tuning (ท่า Resume) ---
print(f"🔥 เริ่มต้นกระบวนการ Fine-Tuning (ตั้งแต่ {start_epoch + 1}/{EPOCHS})...")
history = model.fit(
    train_generator,
    epochs=EPOCHS, # ส่งเป้าหมายรวม
    initial_epoch=start_epoch, # ส่งจุดเริ่มต้นให้มันนับต่อ
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, SaveHistory()],
    verbose=1
)

# --- 9. สร้างและบันทึกกราฟอัตโนมัติ ---
print("\n📊 กำลังวาดกราฟสรุปผลการ Fine-Tune ทั้งหมด...")
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'rb') as f:
        full_history = pickle.load(f)

    acc = full_history['accuracy']
    val_acc = full_history['val_accuracy']
    loss = full_history['loss']
    val_loss = full_history['val_loss']
    epochs_range = range(1, len(acc) + 1) # แก้ให้จุดกราฟเริ่มจาก Epoch 1

    plt.figure(figsize=(12, 5))

    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Fine-Tuning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    # บังคับให้แกน X โชว์เฉพาะจำนวนเต็ม
    plt.xticks(epochs_range)

    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Fine-Tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(epochs_range)

    # บันทึกรูปภาพ
    plt.tight_layout()
    report_path = './models/finetune_report.png'
    plt.savefig(report_path)
    print(f"✅ สร้างกราฟสำเร็จ! เซฟรูปไว้ที่ {report_path} เรียบร้อยแล้วครับ")
else:
    print("❌ ไม่พบไฟล์ประวัติการเทรนสำหรับวาดกราฟ")