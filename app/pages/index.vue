<script setup>
const file = ref(null)
const preview = ref(null)

function onFileSelected(event) {
  const selected = event.target.files[0]
  file.value = selected
  if (selected.type.startsWith('image/')) {
    preview.value = URL.createObjectURL(selected)
  }
}

const loading = ref(false)
const result = ref(null)

async function analyze() {
  loading.value = true
  await new Promise(r => setTimeout(r, 2000))
  result.value = { label: 'FAKE', confidence: 94 }
  loading.value = false
}
</script>

<template>
  <div>

    <!-- 1. HERO SECTION -->
    <section class="min-h-screen flex flex-col items-center justify-center text-center px-6 bg-[#080b10]">
      <span class="text-[#00e5ff] text-xs tracking-widest uppercase border border-[#00e5ff33] px-4 py-2 mb-8">
        AI-Powered Media Verification
      </span>
      <h1 class="text-7xl font-black text-white mb-6 leading-none">
        Detect the <br />
        <span class="text-transparent" style="-webkit-text-stroke: 1px #00e5ff">Unreal</span>
      </h1>
      <p class="text-[#5a7a94] text-sm max-w-md mb-10 leading-relaxed">
        Upload an image or video clip. Our AI analyzes it and tells you if it's a deepfake.
      </p>
      <a href="#detect" class="bg-[#00e5ff] text-[#080b10] px-8 py-3 text-sm font-bold tracking-widest uppercase hover:shadow-[0_0_24px_#00e5ff88] transition-all">
        Try It Now
      </a>
    </section>

    <!-- 2. UPLOAD SECTION -->
    <section id="detect" class="bg-[#080b10] px-12 py-24">
      <p class="text-[#00e5ff] text-xs tracking-widest uppercase mb-2">// analysis engine</p>
      <h2 class="text-4xl font-black text-white mb-4">Upload & Analyze</h2>
      <p class="text-[#5a7a94] text-sm mb-12">Drop an image or video. The AI will return a deepfake probability score.</p>

      <div class="grid grid-cols-2 gap-6">

        <!-- คอลัมน์ซ้าย -->
        <div>
          <div
            class="border border-dashed border-[#1e2d3d] bg-[#0d1117] rounded p-12 flex flex-col items-center justify-center min-h-64 cursor-pointer hover:border-[#00e5ff] transition-all"
            @click="$refs.fileInput.click()"
          >
            <img v-if="preview" :src="preview" class="max-h-48 mb-4 rounded" />
            <template v-else>
              <p class="text-4xl mb-4">📁</p>
              <p class="text-white font-bold mb-2">Drop file here</p>
              <p class="text-[#5a7a94] text-sm">or click to browse</p>
              <div class="flex gap-2 mt-6">
                <span class="text-xs border border-[#1e2d3d] text-[#5a7a94] px-2 py-1">JPG</span>
                <span class="text-xs border border-[#1e2d3d] text-[#5a7a94] px-2 py-1">PNG</span>
                <span class="text-xs border border-[#1e2d3d] text-[#5a7a94] px-2 py-1">MP4</span>
              </div>
            </template>
            <input ref="fileInput" type="file" accept="image/*,video/mp4" class="hidden" @change="onFileSelected" />
          </div>

          <button
            v-if="file"
            @click.stop="analyze"
            class="mt-4 w-full bg-[#00e5ff] text-[#080b10] py-3 text-sm font-bold tracking-widest uppercase hover:shadow-[0_0_24px_#00e5ff88] transition-all"
          >
            {{ loading ? 'Analyzing...' : 'Analyze for Deepfake' }}
          </button>
        </div>

        <!-- คอลัมน์ขวา -->
        <div class="border border-[#1e2d3d] bg-[#0d1117] rounded p-8 min-h-64 flex flex-col">
          <p class="text-white font-bold mb-6">Analysis Result</p>

          <div v-if="loading" class="flex-1 flex flex-col items-center justify-center gap-4">
            <div class="w-8 h-8 border-2 border-[#00e5ff] border-t-transparent rounded-full animate-spin"></div>
            <p class="text-[#5a7a94] text-sm">Analyzing...</p>
          </div>

          <div v-else-if="!result" class="flex-1 flex flex-col items-center justify-center text-[#5a7a94] text-sm">
            <p class="text-3xl mb-4">🔍</p>
            <p>Upload a file to begin analysis</p>
          </div>

          <div v-else class="flex-1 flex flex-col gap-6">
            <div
              class="px-4 py-3 font-black text-xl"
              :class="result.label === 'FAKE'
                ? 'bg-red-500/10 border border-red-500/40 text-red-400'
                : 'bg-green-500/10 border border-green-500/40 text-green-400'"
            >
              {{ result.label === 'FAKE' ? '⚠ Deepfake Detected' : '✓ Likely Authentic' }}
            </div>
            <div>
              <div class="flex justify-between text-xs text-[#5a7a94] mb-2">
                <span>Confidence</span>
                <span>{{ result.confidence }}%</span>
              </div>
              <div class="h-1.5 bg-[#1e2d3d] rounded">
                <div
                  class="h-full rounded transition-all duration-1000"
                  :class="result.label === 'FAKE' ? 'bg-red-400' : 'bg-green-400'"
                  :style="{ width: result.confidence + '%' }"
                ></div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </section>

    <!-- 3. HOW IT WORKS SECTION -->
    <section class="bg-[#080b10] border-t border-[#1e2d3d] px-12 py-24">
      <p class="text-[#00e5ff] text-xs tracking-widest uppercase mb-2">// methodology</p>
      <h2 class="text-4xl font-black text-white mb-12">How It Works</h2>

      <div class="grid grid-cols-4 gap-px bg-[#1e2d3d]">
        <div class="bg-[#080b10] p-8">
          <p class="text-[#00e5ff] text-4xl font-black mb-4">01</p>
          <p class="text-white font-bold mb-2">Upload</p>
          <p class="text-[#5a7a94] text-sm leading-relaxed">อัปโหลดรูปภาพหรือวิดีโอที่ต้องการตรวจสอบ</p>
        </div>
        <div class="bg-[#080b10] p-8">
          <p class="text-[#00e5ff] text-4xl font-black mb-4">02</p>
          <p class="text-white font-bold mb-2">Analyze</p>
          <p class="text-[#5a7a94] text-sm leading-relaxed">AI วิเคราะห์ใบหน้า texture และ pattern ที่ผิดปกติ</p>
        </div>
        <div class="bg-[#080b10] p-8">
          <p class="text-[#00e5ff] text-4xl font-black mb-4">03</p>
          <p class="text-white font-bold mb-2">Detect</p>
          <p class="text-[#5a7a94] text-sm leading-relaxed">โมเดลคำนวณความน่าจะเป็นว่าเป็น deepfake</p>
        </div>
        <div class="bg-[#080b10] p-8">
          <p class="text-[#00e5ff] text-4xl font-black mb-4">04</p>
          <p class="text-white font-bold mb-2">Result</p>
          <p class="text-[#5a7a94] text-sm leading-relaxed">แสดงผลพร้อม confidence score และรายละเอียด</p>
        </div>
      </div>
    </section>

<!-- 4. FOOTER -->
<footer class="bg-[#080b10] border-t border-[#1e2d3d] px-12 py-8 flex justify-between items-center">
  <span class="text-white font-bold">TruthLens</span>
  <span class="text-[#5a7a94] text-sm">Built with Nuxt 4 + AI Model</span>
  <span class="text-[#5a7a94] text-sm">© 2026</span>
</footer>

  </div>
</template>