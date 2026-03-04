// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-03-04',

  future: {
    compatibilityVersion: 4
  },

  devtools: { enabled: true },
  modules: ['@nuxtjs/tailwindcss']
})