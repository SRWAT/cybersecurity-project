export default defineEventHandler(async (event) => {
  const form = await readFormData(event)
  const file = form.get('file') as File

  if (!file) {
    throw createError({ statusCode: 400, message: 'No file provided' })
  }

  // เช็คว่าเป็นรูป วิดีโอ หรือเสียง เพื่อเลือกประตูเข้า API ให้ถูกช่อง
// เปลี่ยนจาก localhost เป็นลิงก์ที่ได้จาก Ngrok
let apiUrl = 'https://1a2b-34-56.ngrok-free.app/predict' 

if (file.type.startsWith('video/')) {
  apiUrl = 'https://1a2b-34-56.ngrok-free.app/predict/video'
} else if (file.type.startsWith('audio/')) {
  apiUrl = 'https://1a2b-34-56.ngrok-free.app/predict/audio'
}
  // ยิงไปหา AI ของ Mo
  const result = await $fetch(apiUrl, { 
    method: 'POST', 
    body: form 
  })

  // ส่งผลลัพธ์กลับไปให้หน้าเว็บแสดงผล
  return result
})