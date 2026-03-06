import os
import shutil
import random
from pathlib import Path

# --- ตั้งค่าตำแหน่งโฟลเดอร์ ---
# อ้างอิงจากรูป VS Code ของ Mo
BASE_DIR = './data'
SOURCE_DIR = Path(BASE_DIR) 
TRAIN_DIR = Path(f'{BASE_DIR}/train')
VAL_DIR = Path(f'{BASE_DIR}/val')

# สัดส่วนการแบ่งข้อมูล
TRAIN_SPLIT = 0.8  # 80% สำหรับเทรน

def setup_directories():
    """สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี"""
    for folder in [TRAIN_DIR, VAL_DIR]:
        for category in ['real', 'fake']:
            os.makedirs(folder / category, exist_ok=True)
    print("✅ สร้างโครงสร้างโฟลเดอร์ Train และ Val เรียบร้อยแล้ว")

def process_category(category_name):
    """จัดการดึงไฟล์ เปลี่ยนชื่อ และแบ่งหมวดหมู่"""
    print(f"กำลังเริ่มจัดการข้อมูลหมวด: {category_name}...")
    
    # ดึงไฟล์จาก data/real หรือ data/fake โดยตรง
    raw_path = SOURCE_DIR / category_name
    
    # ตรวจสอบว่าโฟลเดอร์ต้นทางมีอยู่จริงไหม
    if not raw_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์: {raw_path} กรุณาตรวจสอบตำแหน่งไฟล์อีกครั้ง")
        return

    # ดึงไฟล์ .png ทั้งหมดจากโฟลเดอร์ย่อยทุกชั้น
    all_images = list(raw_path.rglob('*.png'))
    
    if len(all_images) == 0:
        print(f"⚠️ ไม่พบไฟล์ .png ใน {raw_path}")
        return

    print(f"  🔍 พบรูปภาพทั้งหมด {len(all_images):,} รูป")
    
    # สุ่มลำดับภาพ
    random.shuffle(all_images)
    
    # แบ่ง 80/20
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    train_set = all_images[:split_idx]
    val_set = all_images[split_idx:]
    
    def move_and_rename(image_list, target_base_path):
        for idx, old_path in enumerate(image_list):
            # สร้างชื่อไฟล์ใหม่กันซ้ำ: หมวด_ชื่อโฟลเดอร์ย่อย_ชื่อไฟล์เดิม.png
            subfolder_name = old_path.parent.name
            new_name = f"{category_name}_{subfolder_name}_{old_path.name}"
            dest_path = target_base_path / category_name / new_name
            
            # คัดลอกไฟล์
            shutil.copy2(old_path, dest_path)
            
            if (idx + 1) % 10000 == 0:
                print(f"    ...จัดการไปแล้ว {idx + 1:,} รูป")

    print(f"  🚀 กำลังคัดลอกรูปไปที่ Train ({len(train_set):,} รูป)...")
    move_and_rename(train_set, TRAIN_DIR)
    
    print(f"  🚀 กำลังคัดลอกรูปไปที่ Validation ({len(val_set):,} รูป)...")
    move_and_rename(val_set, VAL_DIR)
    print(f"✅ จัดการหมวด {category_name} สำเร็จ!\n")

if __name__ == "__main__":
    print("--- Deep Learning Data Preparation Started ---")
    setup_directories()
    
    # เริ่มทำหมวด Real และ Fake
    process_category('real')
    process_category('fake')
    
    print("🎉 ข้อมูลพร้อมสำหรับการ Train แล้วครับ Mo!")