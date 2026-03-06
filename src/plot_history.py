import matplotlib.pyplot as plt
import pickle
import os

# --- 1. โหลดข้อมูลประวัติการเทรน ---
history_path = './models/train_history.pkl'
if not os.path.exists(history_path):
    print("❌ ไม่พบไฟล์ประวัติการเทรน! กรุณารอให้เทรนเสร็จก่อนนะครับ")
    exit()

with open(history_path, 'rb') as f:
    history = pickle.load(f)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs_range = range(len(acc))

# --- 2. วาดกราฟ ---
plt.figure(figsize=(12, 5))

# กราฟ Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# กราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

# เซฟรูปเก็บไว้ใส่รายงาน
plt.tight_layout()
plt.savefig('./models/training_report.png')
print("✅ สร้างกราฟสำเร็จ! ดูรูปได้ที่ models/training_report.png")
plt.show()