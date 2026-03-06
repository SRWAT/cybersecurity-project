import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pickle

# --- 1. การตั้งค่าสเปก ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 16 
EPOCHS = 19 # จำนวน Epoch ทั้งหมดที่อยากให้ไปถึง
MODEL_PATH = './models/best_xception_model.h5'
HISTORY_PATH = './models/train_history.pkl'

# --- 2. เตรียมท่อส่งข้อมูล ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './data/train/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    './data/val/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 3. เช็คประวัติการเทรนเดิม เพื่อกำหนดจุดเริ่มนับ Epoch ---
initial_epoch = 0
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'rb') as f:
        existing_history = pickle.load(f)
        # นับว่าในประวัติเดิมมีข้อมูลอยู่กี่ Epoch แล้ว
        initial_epoch = len(existing_history.get('accuracy', []))
    print(f"📈 Mo! ตรวจพบประวัติเดิม เทรนไปแล้ว {initial_epoch} รอบ จะเริ่มต่อที่ Epoch {initial_epoch + 1} ครับ")

# --- 4. โหลดหรือสร้างโมเดล ---
if os.path.exists(MODEL_PATH):
    print(f"📦 กำลังโหลดโมเดลเดิมจาก {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)
else:
    print("🆕 เริ่มสร้างโมเดลใหม่...")
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(1, activation='sigmoid') 
    ])

# --- 5. Compile โมเดล ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
                current_history[key].append(logs[key])
        else:
            current_history = {key: [val] for key, val in logs.items()}
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(current_history, f)

# --- 7. Callbacks มาตรฐาน ---
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- 8. เริ่มเทรน (ใส่ initial_epoch เข้าไป) ---
print(f"🚀 เริ่มการเรียนรู้ต่อจาก Epochที่ {initial_epoch + 1}...")
history = model.fit(
    train_generator,
    epochs=EPOCHS, # จะวิ่งไปจนถึง Epoch ที่ 20
    initial_epoch=initial_epoch, # จุดเริ่มต้น (เช่น ถ้าเคยทำไป 2 มันจะเริ่มที่ 3)
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, SaveHistory()],
    verbose=1
)