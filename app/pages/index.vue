<script setup>
import { ref, onBeforeUnmount, nextTick } from 'vue'
import WaveSurfer from 'wavesurfer.js'

useHead({ 
  title: 'Deepfake Detection Engine',
  meta: [
    { name: 'description', content: 'Advanced Neural Network Engine for Deepfake Detection' }
  ],
  link: [
    { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }
  ]
})

const file = ref(null)
const preview = ref(null)
const fileType = ref('')

const isCameraActive = ref(false)
const videoRef = ref(null)
const canvasRef = ref(null)
let mediaStream = null

const waveformRef = ref(null)
let wavesurfer = null

const loadingLogs = ref([])
const logMessages = [
  "Initializing AI Engine...",
  "Loading XceptionNet Weights...",
  "Extracting visual frames...",
  "Running facial bounding box scanner...",
  "Analyzing pixel-level artifacts...",
  "Compiling threat report..."
]
let logInterval = null

function startLoadingLogs() {
  loadingLogs.value = []
  let i = 0
  loadingLogs.value.push(`> ${logMessages[i]}`)
  
  logInterval = setInterval(() => {
    i++
    if (i < logMessages.length) {
      loadingLogs.value.push(`> ${logMessages[i]}`)
    } else {
      clearInterval(logInterval)
    }
  }, 800) 
}

function stopLoadingLogs() {
  clearInterval(logInterval)
  loadingLogs.value = []
}

function destroyWaveform() {
  if (wavesurfer) {
    wavesurfer.destroy()
    wavesurfer = null
  }
}

async function initWaveform(url) {
  await nextTick()
  if (!waveformRef.value) return
  
  wavesurfer = WaveSurfer.create({
    container: waveformRef.value,
    waveColor: 'rgba(0, 229, 255, 0.3)',
    progressColor: '#00e5ff',
    cursorWidth: 2,
    cursorColor: '#ef4444',
    barWidth: 2,
    barGap: 2,
    barRadius: 2,
    height: 60,
    interact: false,
  })
  
  wavesurfer.on('ready', () => {
    wavesurfer.setVolume(0)
    wavesurfer.play()
  })

  wavesurfer.load(url)
}

async function onFileSelected(event) {
  const selected = event.target.files[0]
  if (!selected) return

  file.value = selected
  preview.value = URL.createObjectURL(selected)
  destroyWaveform()

  if (selected.type.startsWith('image/')) {
    fileType.value = 'image'
  } else if (selected.type.startsWith('video/')) {
    fileType.value = 'video'
    initWaveform(preview.value)
  } else if (selected.type.startsWith('audio/')) {
    fileType.value = 'audio'
    initWaveform(preview.value)
  }
}

async function startCamera() {
  isCameraActive.value = true
  file.value = null
  preview.value = null
  destroyWaveform()
  
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true })
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream
    }
  } catch (error) {
    console.error("Camera error:", error)
    alert("ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบสิทธิ์การอนุญาตในเบราว์เซอร์ครับ")
    isCameraActive.value = false
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop())
    mediaStream = null
  }
  isCameraActive.value = false
}

function capturePhoto() {
  if (!videoRef.value || !canvasRef.value) return

  const video = videoRef.value
  const canvas = canvasRef.value
  
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  const ctx = canvas.getContext('2d')
  ctx.translate(canvas.width, 0)
  ctx.scale(-1, 1)
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

  canvas.toBlob((blob) => {
    if (!blob) return
    const capturedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" })
    file.value = capturedFile
    preview.value = URL.createObjectURL(capturedFile)
    fileType.value = 'image'
    stopCamera() 
  }, 'image/jpeg', 0.95)
}

onBeforeUnmount(() => {
  stopCamera()
  destroyWaveform()
  stopLoadingLogs()
})

const loading = ref(false)
const result = ref(null)

async function analyze() {
  if (!file.value) return
  
  loading.value = true
  result.value = null
  startLoadingLogs() 

  const form = new FormData()
  form.append('file', file.value)

  try {
    const data = await $fetch('/api/predict', {
      method: 'POST',
      body: form
    })
    result.value = data
  } catch (error) {
    console.error("Error analyzing file:", error)
    alert("เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบ AI")
  } finally {
    loading.value = false
    stopLoadingLogs() 
  }
}

function clearFile() {
  file.value = null
  preview.value = null
  destroyWaveform()
}
</script>

<template>
  <div class="font-sans bg-[#05080c] text-white selection:bg-[#00e5ff] selection:text-black">
    
    <section class="min-h-[85vh] flex flex-col items-center justify-center text-center px-6 relative overflow-hidden">
      <div class="absolute inset-0 opacity-10 pointer-events-none" style="background-image: linear-gradient(#1e2d3d 1px, transparent 1px), linear-gradient(90deg, #1e2d3d 1px, transparent 1px); background-size: 40px 40px;"></div>
      
      <div class="flex items-center gap-3 mb-8 border border-[#00e5ff33] bg-[#00e5ff]/5 px-4 py-2 shadow-[0_0_15px_#00e5ff33] backdrop-blur-sm z-10">
        <span class="w-2 h-2 rounded-full bg-[#00e5ff] animate-pulse"></span>
        <span class="text-[#00e5ff] text-xs tracking-widest uppercase font-bold">Deepfake Detection Engine</span>
      </div>
      
      <h1 class="text-6xl md:text-8xl font-black text-white mb-6 leading-none tracking-tight z-10">
        Detect the <br />
        <span class="text-transparent" style="-webkit-text-stroke: 1px #00e5ff">Unreal</span>
      </h1>
      <p class="text-[#5a7a94] text-sm md:text-base max-w-xl mb-12 leading-relaxed z-10">
        Advanced Neural Network Engine for Deepfake Detection. Identify synthetic media manipulations in images, videos, and audio streams with high precision.
      </p>
      <a href="#detect" class="bg-[#00e5ff] text-[#080b10] px-10 py-4 text-sm font-black tracking-widest uppercase hover:shadow-[0_0_30px_#00e5ff88] transition-all z-10">
        Initialize Scanner
      </a>
    </section>

    <section class="bg-[#080b10] border-t border-[#1e2d3d] px-8 md:px-12 py-16">
      <div class="max-w-6xl mx-auto">
        <p class="text-[#00e5ff] text-xs tracking-widest uppercase mb-2">// STANDARD OPERATING PROCEDURE</p>
        <h2 class="text-2xl font-black text-white mb-10">HOW TO USE</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="border border-[#1e2d3d] bg-[#0d1117] p-6 rounded hover:border-[#00e5ff]/50 transition-colors">
            <div class="text-[#00e5ff] text-3xl mb-4 font-black">01</div>
            <h3 class="text-white font-bold mb-2 uppercase text-sm tracking-widest">Input Media</h3>
            <p class="text-[#5a7a94] text-xs leading-relaxed">Upload a target file (JPG, PNG, MP4, WAV) or initialize the live camera module to capture an image directly.</p>
          </div>
          <div class="border border-[#1e2d3d] bg-[#0d1117] p-6 rounded hover:border-[#00e5ff]/50 transition-colors">
            <div class="text-[#00e5ff] text-3xl mb-4 font-black">02</div>
            <h3 class="text-white font-bold mb-2 uppercase text-sm tracking-widest">Execute Scan</h3>
            <p class="text-[#5a7a94] text-xs leading-relaxed">The system automatically detects the file type and routes it through the appropriate XceptionNet or Wav2Vec models.</p>
          </div>
          <div class="border border-[#1e2d3d] bg-[#0d1117] p-6 rounded hover:border-[#00e5ff]/50 transition-colors">
            <div class="text-[#00e5ff] text-3xl mb-4 font-black">03</div>
            <h3 class="text-white font-bold mb-2 uppercase text-sm tracking-widest">Review Report</h3>
            <p class="text-[#5a7a94] text-xs leading-relaxed">Analyze the extracted bounding boxes, overall confidence scores, and format-specific technical logs.</p>
          </div>
        </div>
      </div>
    </section>

    <section id="detect" class="bg-[#05080c] px-8 md:px-12 py-20 border-t border-[#1e2d3d]">
      <div class="max-w-7xl mx-auto">
        <div class="flex justify-between items-end mb-8">
          <div>
            <p class="text-[#00e5ff] text-xs tracking-widest uppercase mb-2">// CORE ANALYSIS MODULE</p>
            <h2 class="text-4xl font-black text-white">Threat Detection</h2>
          </div>
          <div class="hidden md:flex items-center gap-2 text-[#5a7a94] text-[10px] uppercase tracking-widest border border-[#1e2d3d] px-3 py-1.5 bg-[#080b10] rounded">
            <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
            <span>Strict Privacy: Media processed locally and auto-purged.</span>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          <div class="flex flex-col gap-4">
            <div
              class="border border-dashed border-[#1e2d3d] bg-[#0d1117] rounded p-8 flex flex-col items-center justify-center min-h-[400px] transition-all relative overflow-hidden h-full"
              :class="!isCameraActive ? 'cursor-pointer hover:border-[#00e5ff]' : ''"
              @click="!isCameraActive && !file && $refs.fileInput.click()"
            >
              
              <template v-if="isCameraActive">
                <div class="w-full flex flex-col items-center z-20">
                  <video ref="videoRef" autoplay playsinline class="w-full aspect-video mb-4 rounded border border-[#00e5ff] shadow-[0_0_15px_#00e5ff55] bg-black transform scale-x-[-1] object-cover"></video>
                  <canvas ref="canvasRef" class="hidden"></canvas>
                  
                  <div class="flex gap-4 mt-2">
                    <button @click.stop="capturePhoto" class="bg-[#00e5ff] text-[#080b10] px-8 py-2.5 text-xs font-black uppercase tracking-widest hover:shadow-[0_0_15px_#00e5ff88] transition-all flex items-center gap-2">
                      <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                      Capture Frame
                    </button>
                    <button @click.stop="stopCamera" class="border border-red-500 text-red-500 px-6 py-2.5 text-xs font-bold uppercase tracking-widest hover:bg-red-500 hover:text-white transition-all">
                      Abort
                    </button>
                  </div>
                </div>
              </template>

              <template v-else-if="file">
                <img v-if="fileType === 'image'" :src="preview" class="max-h-64 object-contain mb-4 rounded border border-[#1e2d3d] shadow-lg z-10" />
                <video v-else-if="fileType === 'video'" :src="preview" controls class="w-full aspect-video mb-4 rounded border border-[#1e2d3d] shadow-lg z-10 bg-black"></video>
                <audio v-else-if="fileType === 'audio'" :src="preview" controls class="mb-4 w-full z-10"></audio>
                
                <div v-show="['video', 'audio'].includes(fileType)" class="w-full bg-[#05080c] border border-[#1e2d3d] rounded p-3 mb-6 relative overflow-hidden z-10 shadow-inner">
                  <div class="absolute top-0 left-0 w-1 h-full bg-[#00e5ff] rounded-l"></div>
                  <p class="text-[#00e5ff] text-[9px] font-bold uppercase tracking-widest mb-2 ml-2 flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
                    Audio Frequency Scanner
                  </p>
                  <div ref="waveformRef" class="w-full h-[60px]"></div>
                </div>
                
                <p class="text-[#00e5ff] text-xs text-center bg-[#1e2d3d] px-4 py-1.5 rounded z-10 mb-6 font-mono">{{ file.name }}</p>
                
                <div class="flex gap-3 z-20">
                  <button @click.stop="$refs.fileInput.click()" class="border border-[#1e2d3d] text-[#5a7a94] px-5 py-2 text-xs font-bold uppercase hover:border-[#00e5ff] hover:text-[#00e5ff] transition-all bg-[#080b10]">
                    Replace
                  </button>
                  <button @click.stop="clearFile" class="border border-red-500/30 text-red-400 px-5 py-2 text-xs font-bold uppercase hover:bg-red-500 hover:text-white transition-all bg-[#080b10]">
                    Clear
                  </button>
                </div>
              </template>

              <template v-else>
                <div class="w-16 h-16 rounded-full bg-[#1e2d3d] flex items-center justify-center mb-6 border border-[#5a7a94]/30">
                  <svg class="w-8 h-8 text-[#5a7a94]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" /></svg>
                </div>
                <p class="text-white font-bold mb-2 text-lg">Drop media payload here</p>
                <p class="text-[#5a7a94] text-sm mb-8">or click to browse local directories</p>
                
                <div class="flex items-center gap-4 w-full max-w-xs mb-8">
                  <div class="h-px bg-[#1e2d3d] flex-1"></div>
                  <span class="text-[#5a7a94] text-xs font-bold uppercase">OR</span>
                  <div class="h-px bg-[#1e2d3d] flex-1"></div>
                </div>

                <button @click.stop="startCamera" class="bg-[#1e2d3d] text-white px-6 py-3 rounded text-xs font-bold uppercase tracking-widest hover:bg-[#00e5ff] hover:text-[#080b10] transition-all flex items-center gap-2 border border-[#5a7a94]/30">
                  <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                  Initialize Live Camera
                </button>

                <div class="flex gap-2 mt-10 flex-wrap justify-center opacity-70">
                  <span class="text-[10px] uppercase font-bold border border-[#1e2d3d] text-[#5a7a94] px-2 py-1 bg-[#080b10]">JPG</span>
                  <span class="text-[10px] uppercase font-bold border border-[#1e2d3d] text-[#5a7a94] px-2 py-1 bg-[#080b10]">PNG</span>
                  <span class="text-[10px] uppercase font-bold border border-[#1e2d3d] text-[#5a7a94] px-2 py-1 bg-[#080b10]">MP4</span>
                  <span class="text-[10px] uppercase font-bold border border-[#1e2d3d] text-[#5a7a94] px-2 py-1 bg-[#080b10]">WAV</span>
                </div>
              </template>

              <input ref="fileInput" type="file" accept="image/*,video/mp4,audio/*" class="hidden" @change="onFileSelected" />
            </div>

            <button
              v-if="file"
              @click.stop="analyze"
              class="w-full bg-[#00e5ff] text-[#080b10] py-4 text-sm font-black tracking-widest uppercase hover:shadow-[0_0_24px_#00e5ff88] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="loading"
            >
              {{ loading ? 'Processing Neural Networks...' : 'Execute Analysis' }}
            </button>
          </div>

          <div class="border border-[#1e2d3d] bg-[#0d1117] rounded flex flex-col relative overflow-hidden shadow-xl">
            
            <div class="bg-[#1e2d3d]/40 px-6 py-4 border-b border-[#1e2d3d] flex justify-between items-center">
              <p class="text-white font-bold tracking-widest uppercase text-sm flex items-center gap-2">
                <svg class="w-5 h-5 text-[#00e5ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                Analysis Report
              </p>
              <span v-if="result" class="text-[10px] font-black px-2 py-1 bg-[#00e5ff]/10 text-[#00e5ff] border border-[#00e5ff]/30 rounded tracking-widest">COMPLETED</span>
              <span v-else-if="loading" class="text-[10px] font-black px-2 py-1 bg-yellow-500/10 text-yellow-500 border border-yellow-500/30 rounded animate-pulse tracking-widest">PROCESSING</span>
              <span v-else class="text-[10px] font-black px-2 py-1 bg-[#1e2d3d] text-[#5a7a94] border border-[#1e2d3d] rounded tracking-widest">STANDBY</span>
            </div>

            <div class="p-6 md:p-8 flex-1 flex flex-col bg-[#080b10]/50">
              
              <div v-if="loading" class="flex-1 flex flex-col items-center justify-center gap-6">
                <div class="w-full bg-[#05080c] border border-[#1e2d3d] rounded p-4 h-48 flex flex-col justify-end overflow-hidden relative">
                  <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[#00e5ff] to-transparent opacity-50"></div>
                  <div class="flex flex-col gap-2">
                    <p v-for="(log, idx) in loadingLogs" :key="idx" class="text-[#00e5ff] text-[10px] font-mono uppercase tracking-widest animate-pulse flex items-center gap-2">
                      <span class="text-[#00e5ff] font-bold">></span>
                      {{ log }}
                    </p>
                  </div>
                </div>
              </div>

              <div v-else-if="!result" class="flex-1 flex flex-col items-center justify-center text-[#5a7a94] text-sm">
                <div class="w-20 h-20 rounded-full border border-dashed border-[#5a7a94]/50 flex items-center justify-center mb-6">
                  <svg class="w-10 h-10 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                </div>
                <p class="font-mono text-xs uppercase tracking-widest">System ready. Awaiting telemetry.</p>
              </div>

              <div v-else class="flex-1 flex flex-col gap-6">
                
                <div v-if="result.frames && result.frames.length > 0" class="bg-[#05080c] border border-[#1e2d3d] rounded p-4 overflow-hidden">
                  <p class="text-[#00e5ff] text-[10px] font-bold uppercase tracking-widest mb-3 border-b border-[#1e2d3d] pb-2 flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" /></svg>
                    Visual Extraction Data
                  </p>
                  <div class="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-[#00e5ff] scrollbar-track-[#1e2d3d]">
                    <img v-for="(frame, index) in result.frames" :key="index" :src="frame" class="h-24 md:h-32 rounded border border-[#1e2d3d] hover:border-[#00e5ff] transition-all object-cover shadow-md" />
                  </div>
                </div>

                <div
                  class="px-6 py-5 flex items-center gap-5 rounded border shadow-lg"
                  :class="(result.prediction || result.visual_prediction || result.label) === 'FAKE'
                    ? 'bg-red-500/10 border-red-500/40'
                    : 'bg-green-500/10 border-green-500/40'"
                >
                  <div class="text-4xl md:text-5xl drop-shadow-lg">
                    <svg v-if="(result.prediction || result.visual_prediction || result.label) === 'FAKE'" class="w-12 h-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                    <svg v-else class="w-12 h-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0112 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 01-3.296-1.043 3.745 3.745 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.746 3.746 0 013.296-1.043A3.746 3.746 0 0112 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z" /></svg>
                  </div>
                  <div>
                    <p class="text-[10px] font-bold uppercase tracking-widest mb-1" :class="(result.prediction || result.visual_prediction || result.label) === 'FAKE' ? 'text-red-400' : 'text-green-400'">System Verdict</p>
                    <p class="text-2xl md:text-3xl font-black text-white tracking-tight">
                      {{ (result.prediction || result.visual_prediction || result.label) === 'FAKE' ? 'Deepfake Detected' : 'Authentic Media' }}
                    </p>
                  </div>
                </div>
                
                <div>
                  <div class="flex justify-between text-[10px] font-bold text-[#5a7a94] mb-2 uppercase tracking-widest">
                    <span>Overall Confidence Level</span>
                    <span class="text-white">{{ result.confidence || result.visual_confidence || 0 }}%</span>
                  </div>
                  <div class="h-2.5 bg-[#1e2d3d] rounded overflow-hidden">
                    <div
                      class="h-full rounded transition-all duration-1000"
                      :class="(result.prediction || result.visual_prediction || result.label) === 'FAKE' ? 'bg-red-500 shadow-[0_0_15px_#ef4444]' : 'bg-green-500 shadow-[0_0_15px_#22c55e]'"
                      :style="{ width: (result.confidence || result.visual_confidence || 0) + '%' }"
                    ></div>
                  </div>
                </div>

                <div class="mt-2 border border-[#1e2d3d] rounded bg-[#0d1117]">
                  <div class="bg-[#1e2d3d]/30 px-5 py-2 text-[10px] font-bold text-white uppercase tracking-widest border-b border-[#1e2d3d] flex items-center gap-2">
                    <svg class="w-4 h-4 text-[#00e5ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
                    Diagnostic Breakdown
                  </div>
                  
                  <div class="p-5 flex flex-col gap-4 text-xs">
                    
                    <div class="flex justify-between items-center border-b border-[#1e2d3d]/50 pb-3">
                      <span class="text-[#5a7a94] uppercase tracking-wider font-bold">Input Modality:</span>
                      <span class="text-[#00e5ff] uppercase font-black bg-[#00e5ff]/10 px-2 py-0.5 rounded">{{ fileType }}</span>
                    </div>

                    <div v-if="['image', 'video'].includes(fileType)" class="flex justify-between items-center border-b border-[#1e2d3d]/50 pb-3">
                      <span class="text-[#5a7a94] uppercase tracking-wider font-bold">Visual Engine (Xception):</span>
                      <span class="font-black px-2 py-0.5 rounded uppercase tracking-wider" :class="(result.prediction || result.visual_prediction || result.label) === 'FAKE' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'">
                        {{ result.prediction || result.visual_prediction || result.label }}
                      </span>
                    </div>

                    <div v-if="['audio', 'video'].includes(fileType)" class="flex justify-between items-center border-b border-[#1e2d3d]/50 pb-3">
                      <span class="text-[#5a7a94] uppercase tracking-wider font-bold">Audio Engine (Wav2Vec):</span>
                      <span class="flex items-center gap-2 font-black uppercase tracking-wider" :class="result.audio_extracted ? 'text-yellow-400' : 'text-[#5a7a94]'">
                        <span v-if="result.audio_extracted" class="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse"></span>
                        {{ result.audio_extracted ? 'Pending Model' : 'N/A' }}
                      </span>
                    </div>

                    <div class="bg-[#05080c] p-3 rounded border border-[#1e2d3d] mt-1 relative">
                      <div class="absolute top-0 left-0 w-1 h-full bg-[#00e5ff] rounded-l"></div>
                      <span class="text-[#5a7a94] text-[9px] block mb-1 uppercase tracking-widest font-bold ml-2">Terminal Output</span>
                      <span class="text-[#00e5ff] text-xs font-mono ml-2 block">> {{ result.message || 'Analysis operation completed successfully.' }}</span>
                    </div>

                  </div>
                </div>

              </div>
            </div>
          </div>

        </div>
      </div>
    </section>

    <footer class="bg-[#05080c] border-t border-[#1e2d3d] flex flex-col md:flex-row justify-between items-center px-8 py-4 gap-4">
      <div class="flex items-center gap-6">
        <span class="text-white font-black uppercase tracking-widest text-sm">Deepfake Detector</span>
      </div>
      
      <div class="flex gap-4 text-[10px] font-mono uppercase tracking-widest border border-[#1e2d3d] bg-[#080b10] px-3 py-1.5 rounded text-[#5a7a94]">
        <span class="flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span> API Online</span>
        <span class="border-l border-[#1e2d3d] pl-4 flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span> GPU Standby</span>
      </div>
    </footer>

  </div>
</template>