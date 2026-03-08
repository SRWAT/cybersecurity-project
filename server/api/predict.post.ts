export default defineEventHandler(async (event) => {
  const form = await readFormData(event)
  const file = form.get('file') as File

  if (!file) {
    throw createError({ statusCode: 400, message: 'No file provided' })
  }

  // เช็คว่าเป็นรูป วิดีโอ หรือเสียง เพื่อเลือกประตูเข้า API ให้ถูกช่อง
  let apiUrl = 'http://localhost:8000/predict' // ตั้งค่าเริ่มต้นเป็นช่องทางตรวจรูปภาพ

  if (file.type.startsWith('video/')) {
    apiUrl = 'http://localhost:8000/predict/video'
  } else if (file.type.startsWith('audio/')) {
    apiUrl = 'http://localhost:8000/predict/audio' // เพิ่มเงื่อนไขให้วิ่งไปหา Wav2Vec
  }

  // ยิงไปหา AI ของ Mo
  const result = await $fetch(apiUrl, { 
    method: 'POST', 
    body: form 
  })

  // ส่งผลลัพธ์กลับไปให้หน้าเว็บแสดงผล
  return result
})