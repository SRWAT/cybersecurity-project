import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import sys
import pickle

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE        = (299, 299)
BATCH_SIZE      = 16
EPOCHS          = 10
FREEZE_LAYERS   = 100       # จำนวน layer แรกของ Xception ที่ lock ไว้
LEARNING_RATE   = 1e-5
PATIENCE        = 3         # EarlyStopping patience

MODELS_DIR          = "./models"
FINETUNED_MODEL     = os.path.join(MODELS_DIR, "finetuned_xception_model.h5")
BASE_MODEL          = os.path.join(MODELS_DIR, "best_xception_model.h5")
HISTORY_PATH        = os.path.join(MODELS_DIR, "finetune_history.pkl")
REPORT_PATH         = os.path.join(MODELS_DIR, "finetune_report.png")

TRAIN_DIR = "./data/train/"
VAL_DIR   = "./data/val/"

HISTORY_KEYS = ("accuracy", "val_accuracy", "loss", "val_loss")

# ─── Data Pipeline ────────────────────────────────────────────────────────────

def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    return train_gen, val_gen

# ─── History Helpers ──────────────────────────────────────────────────────────

def load_history() -> dict:
    """โหลด history จาก pickle ถ้ามี ถ้าไม่มีหรือ corrupt ให้คืน dict เปล่า"""
    if not os.path.exists(HISTORY_PATH):
        return {}
    try:
        with open(HISTORY_PATH, "rb") as f:
            data = pickle.load(f)
        # ตรวจว่าเป็น dict และมี key ที่ถูกต้อง
        if not isinstance(data, dict):
            raise ValueError("รูปแบบ history ไม่ถูกต้อง")
        return data
    except Exception as e:
        print(f"⚠️  อ่านไฟล์ประวัติไม่สำเร็จ เริ่มนับใหม่จาก 0 (Error: {e})")
        return {}

def save_history(data: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(data, f)

def get_start_epoch(history: dict) -> int:
    """ดึงจำนวน epoch ที่เคยเทรนไปแล้วจาก history"""
    epochs_done = len(history.get("accuracy", []))
    if epochs_done > 0:
        print(f"🧠 AI จำได้ว่าเคยเรียนไปแล้ว {epochs_done} Epochs!")
        print(f"⏭️  จะเริ่มเทรนต่อที่ Epoch {epochs_done + 1} → เป้าหมาย {EPOCHS}")
    return epochs_done

# ─── Model Setup ──────────────────────────────────────────────────────────────

def load_or_init_model(start_epoch: int) -> tf.keras.Model:
    """โหลดโมเดลที่ fine-tune ค้างไว้ หรือโหลด base model สำหรับเริ่มใหม่"""
    if os.path.exists(FINETUNED_MODEL) and start_epoch > 0:
        print(f"📦 โหลดโมเดล Fine-tune ที่ค้างไว้: {FINETUNED_MODEL}")
        return load_model(FINETUNED_MODEL)

    if not os.path.exists(BASE_MODEL):
        print(f"❌ ไม่พบ Base Model ที่: {BASE_MODEL}")
        sys.exit(1)

    print(f"📦 โหลด Base Model สำหรับ Fine-tune ครั้งแรก: {BASE_MODEL}")
    return load_model(BASE_MODEL)


def configure_layers(model: tf.keras.Model) -> None:
    """ปลดล็อก top layers ของ Xception สำหรับ fine-tuning"""
    try:
        base = model.layers[0]
        base.trainable = True
        for layer in base.layers[:FREEZE_LAYERS]:
            layer.trainable = False
        unlocked = len(base.layers) - FREEZE_LAYERS
        print(f"🔓 ปลดล็อก {unlocked} layers สำหรับ Fine-tune (freeze {FREEZE_LAYERS} layers แรก)")
    except (IndexError, AttributeError):
        # โมเดลที่ถูก flatten แล้ว ไม่มี sub-model layer[0]
        print("⚠️  ข้ามการแช่แข็ง layer (โมเดลถูก compile flat ไปแล้ว)")

# ─── Custom Callback ──────────────────────────────────────────────────────────

class SaveHistory(tf.keras.callbacks.Callback):
    """Append metric ของแต่ละ epoch เข้าไปใน history pickle แบบ incremental"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        current = load_history()
        for key, val in logs.items():
            current.setdefault(key, []).append(val)
        save_history(current)

# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_report(history: dict) -> None:
    """วาดกราฟ Accuracy + Loss จาก full history และบันทึกเป็นรูปภาพ"""
    # ตรวจว่ามีข้อมูลครบก่อนวาด
    missing = [k for k in HISTORY_KEYS if k not in history or not history[k]]
    if missing:
        print(f"❌ ไม่สามารถวาดกราฟได้ ขาดข้อมูล: {missing}")
        return

    acc      = history["accuracy"]
    val_acc  = history["val_accuracy"]
    loss     = history["loss"]
    val_loss = history["val_loss"]
    x        = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(x, acc,     label="Training Accuracy")
    axes[0].plot(x, val_acc, label="Validation Accuracy")
    axes[0].set_title("Fine-Tuning Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)
    axes[0].set_xticks(x)

    # Loss
    axes[1].plot(x, loss,     label="Training Loss")
    axes[1].plot(x, val_loss, label="Validation Loss")
    axes[1].set_title("Fine-Tuning Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    axes[1].set_xticks(x)

    plt.tight_layout()
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.savefig(REPORT_PATH)
    plt.close(fig)   # คืน memory — ไม่ leak ถ้ารันหลายรอบ
    print(f"✅ บันทึกกราฟสำเร็จ: {REPORT_PATH}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. ตรวจสอบ epoch ที่เคยเทรนไปแล้ว
    past_history = load_history()
    start_epoch  = get_start_epoch(past_history)

    if start_epoch >= EPOCHS:
        print(f"🎉 AI เทรนครบ {EPOCHS} Epochs ตามเป้าหมายแล้ว! ไม่ต้องรันต่อ")
        return

    # 2. เตรียม data generators
    train_gen, val_gen = build_generators()

    # 3. โหลดและ configure โมเดล
    model = load_or_init_model(start_epoch)
    configure_layers(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            FINETUNED_MODEL,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        SaveHistory(),
    ]

    # 5. เทรน
    print(f"🔥 เริ่ม Fine-Tuning (Epoch {start_epoch + 1}/{EPOCHS})...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. วาดกราฟจาก full history
    print("\n📊 กำลังวาดกราฟสรุปผล...")
    full_history = load_history()
    plot_report(full_history)


if __name__ == "__main__":
    main()