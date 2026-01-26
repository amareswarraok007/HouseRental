/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        "engine-blue": "#1d4ed8",
        "engine-dark": "#0f172a",
        "engine-card": "#111827",
      },
    },
  },
  plugins: [],
};
