export default defineEventHandler(async (event) => {
  const form = await readFormData(event)
  const file = form.get('file') as File

  if (!file) {
    throw createError({ statusCode: 400, message: 'No file provided' })
  }

  // เช็คว่าเป็นรูปหรือวิดีโอ เพื่อเลือกประตูเข้า API ของ Mo ให้ถูกช่อง
  const isVideo = file.type.startsWith('video/')
  const apiUrl = isVideo 
    ? 'http://localhost:8000/predict/video' 
    : 'http://localhost:8000/predict'

  // ยิงไปหา AI ของ Mo
  const result = await $fetch(apiUrl, { 
    method: 'POST', 
    body: form 
  })

  // ถ้าเป็นวิดีโอ เราอาจจะต้องปรับให้หน้าเว็บรับตัวแปร visual_label แทน
  return result
})