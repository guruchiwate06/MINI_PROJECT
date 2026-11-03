/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        railway: {
          darkest: '#08090c',
          dark: '#0c0e14',
          surface: '#13161f',
          elevated: '#1a1e2b',
          border: '#262c3e',
          steel: '#3a4256',
          muted: '#8b949e',
          text: '#f0f3f6',
        },
        hud: {
          cyan: '#06b6d4',
          amber: '#f59e0b',
          red: '#ef4444',
          emerald: '#10b981',
          violet: '#8b5cf6',
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar': 'radar 4s linear infinite',
      },
      keyframes: {
        radar: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        }
      }
    },
  },
  plugins: [],
}

