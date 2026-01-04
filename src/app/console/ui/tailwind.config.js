/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Nunito', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      colors: {
        balance: {
          balanced: '#10b981',
          warning: '#f59e0b',
          critical: '#ef4444',
          rigid: '#8b5cf6',
          fragmented: '#f97316',
        },
        alignment: {
          light: '#d1fae5',
          dark: '#064e3b',
        },
        displacement: {
          light: '#fee2e2',
          dark: '#7f1d1d',
        },
      },
    },
  },
  plugins: [],
};

