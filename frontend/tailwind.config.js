/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./frontend/index.html",
    "./frontend/src/**/*.{js,ts,jsx,tsx}",
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          DEFAULT: '#FAF5ED',
          50: '#FFFDF9',
          100: '#FAF5ED',
          200: '#F4EBDE',
          300: '#EAD9C4',
        },
        brand: {
          orange: '#FF5500',
          orangeHover: '#E04B00',
          orangeLight: '#FFEFE6',
          black: '#101010',
          white: '#FFFFFF',
        }
      },
      boxShadow: {
        'boxy': '4px 4px 0px #101010',
        'boxy-sm': '2px 2px 0px #101010',
        'boxy-lg': '6px 6px 0px #101010',
        'boxy-orange': '4px 4px 0px #FF5500',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
