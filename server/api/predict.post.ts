export default defineEventHandler(async (event) => {
  // รับไฟล์ที่ส่งมาจาก browser
  const form = await readFormData(event)
  const file = form.get('file')

  // ถ้าไม่มีไฟล์ส่ง error กลับไป
  if (!file) {
    throw createError({ statusCode: 400, message: 'No file provided' })
  }

  // ตอนนี้ยังเป็น mock
  // พอโมเดลเสร็จ เปลี่ยนแค่ตรงนี้บรรทัดเดียว:
  // const result = await $fetch('http://localhost:8000/predict', { method: 'POST', body: form })
  const result = {
    label: 'FAKE',
    confidence: 94
  }

  return result
})
