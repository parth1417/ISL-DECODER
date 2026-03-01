# ISL DECODER

## Project Title
Next-Generation AI-powered Indian Sign Language (ISL) Recognition Application

## Description
This project is an advanced AI-powered progressive web application (PWA) designed to recognize and interpret Indian Sign Language (ISL) in real-time. Unlike traditional translators that use isolated words, this app intelligently tracks continuous hand gestures and sequences to generate complete, context-aware sentences in natural language. It serves as a true communication bridge between the deaf and hearing communities, prioritizing processing efficiency through edge-AI models right inside the browser.

## Technologies Used
- **Frontend Framework:** React + Vite
- **Styling:** Vanilla CSS (Dark mode, Custom Design Tokens, Micro-Animations)
- **Icons:** Lucide React
- **Mobile Containerization:** Capacitor (Android Integration)
- **Core Processing Architecture (Planned):** TensorFlow Lite Web / MediaPipe Tasks-Vision (for WASM-level device tracking)

## Folder Structure
```
/
├── android/             # Capacitor Native Android App Source
├── dist/                # Production build output
├── public/              # Static public assets
├── src/                 # Main Source Code
│   ├── App.css          # Core Application Styling 
│   ├── App.jsx          # Camera Tracking Logic and UI
│   ├── index.css        # Global CSS Tokens and Fonts
│   └── main.jsx         # App Entry Point
├── .env                 # Environment Variables
├── capacitor.config.json# Capacitor Settings
├── index.html           # Root HTML
├── package.json         # NPM Dependencies
├── README.md            # Documentation
└── vite.config.js       # Vite Configuration
```

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/parth1417/ISL-DECODER.git
   cd ISL-DECODER
   ```
2. **Install all Node dependencies:**
   ```bash
   npm install
   ```
3. **Install Capacitor mobile modules (if running on Android):**
   ```bash
   npm install @capacitor/core
   npm install -D @capacitor/cli
   ```

## How to Run

### Run on Browser (Locally)
1. Start the Vite development server:
   ```bash
   npm run dev
   ```
2. Open your browser and navigate to `http://localhost:5173/`. 
*(Note: Ensure you allow camera permissions when prompted).*

### Run on Android (Via Capacitor)
1. Build the production web assets:
   ```bash
   npm run build
   ```
2. Sync the compiled assets with the native Android project:
   ```bash
   npx cap sync android
   ```
3. Open the project in Android Studio (or run directly if an emulator is connected):
   ```bash
   npx cap open android
   ```

---
**Author:** Kanishk Patil
