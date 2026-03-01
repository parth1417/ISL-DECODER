import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Volume2, VolumeX, WifiOff, Sparkles, CheckCircle2 } from 'lucide-react';
import './App.css';

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [isOffline, setIsOffline] = useState(false);
  const [confidence, setConfidence] = useState(0);

  const [currentSentence, setCurrentSentence] = useState('');
  const [refinedGrammar, setRefinedGrammar] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const customAiModelRef = useRef(null);
  const labelsRef = useRef([]);
  const requestRef = useRef(null);
  let lastVideoTime = -1;

  // Initialize Core Edge ML Models Unblocked
  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();

        // Load the Py-trained model.json generated from the pipeline
        try {
          customAiModelRef.current = await tf.loadLayersModel('/models/isl_model/model.json');
          const res = await fetch('/models/isl_model/labels.json');
          labelsRef.current = await res.json();
          setModelLoaded(true);
        } catch (e) {
          console.error("No custom model trained yet. Waiting for python pipeline to finish...", e);
        }

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
      } catch (err) {
        console.error("AI Model Loading Error:", err);
      }
    };
    loadModels();
  }, []);

  const drawLandmarks = (ctx, landmarks) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "#00ff88";
    ctx.strokeStyle = "#00b8ff";
    ctx.lineWidth = 2;
    // draw points and connections
    for (const hand of landmarks) {
      for (const point of hand) {
        ctx.beginPath();
        ctx.arc(point.x * ctx.canvas.width, point.y * ctx.canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
      // Simple wireframe logic
      ctx.beginPath();
      ctx.moveTo(hand[0].x * ctx.canvas.width, hand[0].y * ctx.canvas.height);
      for (let i = 1; i < hand.length; i++) {
        ctx.lineTo(hand[i].x * ctx.canvas.width, hand[i].y * ctx.canvas.height);
      }
      ctx.stroke();
    }
  };

  const predictLoop = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current || !customAiModelRef.current) return;

    const video = videoRef.current;
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      const results = handLandmarkerRef.current.detectForVideo(video, performance.now());

      const ctx = canvasRef.current.getContext('2d');
      // match video sizing
      canvasRef.current.width = video.videoWidth;
      canvasRef.current.height = video.videoHeight;

      if (results.landmarks) {
        drawLandmarks(ctx, results.landmarks);

        if (results.landmarks.length > 0) {
          setIsProcessing(true);

          let lh = new Array(21 * 3).fill(0);
          let rh = new Array(21 * 3).fill(0);

          if (results.landmarks[0]) {
            if (results.handedness[0][0].categoryName === 'Left') {
              lh = results.landmarks[0].flatMap(pt => [pt.x, pt.y, pt.z]);
            } else {
              rh = results.landmarks[0].flatMap(pt => [pt.x, pt.y, pt.z]);
            }
          }
          if (results.landmarks[1]) {
            if (results.handedness[1][0].categoryName === 'Left') {
              lh = results.landmarks[1].flatMap(pt => [pt.x, pt.y, pt.z]);
            } else {
              rh = results.landmarks[1].flatMap(pt => [pt.x, pt.y, pt.z]);
            }
          }

          const tensor = tf.tensor2d([lh.concat(rh)]);
          const prediction = await customAiModelRef.current.predict(tensor).data();
          const maxIdx = prediction.indexOf(Math.max(...prediction));
          const confidenceScore = prediction[maxIdx];

          if (confidenceScore > 0.4) {
            const detectedLabel = labelsRef.current[maxIdx] || 'Recognizing...';
            setCurrentSentence(detectedLabel);
            setConfidence(Math.round(confidenceScore * 100));
            setRefinedGrammar(`Recognized AI Sequence: "${detectedLabel}"`);
          } else {
            setConfidence(0);
            setCurrentSentence('');
            setRefinedGrammar('');
          }
          tensor.dispose();
        } else {
          setIsProcessing(false);
          setConfidence(0);
        }
      }
    }

    if (isCameraActive) {
      requestRef.current = requestAnimationFrame(predictLoop);
    }
  }, [isCameraActive]);

  useEffect(() => {
    if (isCameraActive) {
      requestRef.current = requestAnimationFrame(predictLoop);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [isCameraActive, predictLoop]);

  const toggleCamera = async () => {
    if (isCameraActive) {
      const stream = videoRef.current?.srcObject;
      stream?.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsCameraActive(true);
        };
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
  };

  return (
    <div className="app-container">
      {/* Top Camera View Section */}
      <div className="camera-wrapper">
        <div className="top-bar">
          <div className="glass-panel status-pill">
            <div className={`status-indicator ${!isCameraActive ? 'paused' : ''}`} style={{ backgroundColor: isCameraActive ? 'var(--accent-primary)' : 'var(--danger)' }}></div>
            <span className="gradient-text" style={{ letterSpacing: '1px' }}>
              {isCameraActive
                ? 'AI VISION ACTIVE'
                : (modelLoaded ? 'MODELS LOADED. OFF' : 'LOADING AI...')}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button className={`icon-btn ${isVoiceEnabled ? 'active' : ''}`} onClick={() => setIsVoiceEnabled(!isVoiceEnabled)}>
              {isVoiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>
            <button className="icon-btn" onClick={toggleCamera}>
              {isCameraActive ? <CameraOff size={20} color="var(--danger)" /> : <Camera size={20} />}
            </button>
          </div>
        </div>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-feed"
          style={{ display: isCameraActive ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          className="tracking-canvas"
          style={{ display: isCameraActive ? 'block' : 'none' }}
        />

        {!isCameraActive && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            <CameraOff size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <p style={{ fontFamily: 'Outfit, sans-serif' }}>Tap camera icon to begin AI tracking</p>
          </div>
        )}
      </div>

      {/* Bottom Output Panel */}
      <div className="output-panel">
        <div className="panel-header">
          <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={18} className="gradient-text" />
            Live Translation
          </h2>
          <div className="confidence-bar">
            <span>Accuracy</span>
            <div className="confidence-fill">
              <div className="confidence-level" style={{ width: `${confidence}%` }}></div>
            </div>
          </div>
        </div>

        <div className="live-text-container">
          {!isCameraActive && !currentSentence ? (
            <div style={{ opacity: 0.3 }}>
              <div className="skeleton-text"></div>
              <div className="skeleton-text" style={{ width: '60%' }}></div>
            </div>
          ) : (
            <div className="live-text">
              {!modelLoaded ?
                "Waiting for Custom Neural Network..." :
                (currentSentence || 'Waiting for gestures...')}
              {isProcessing && <span className="cursor" />}
            </div>
          )}
        </div>

        {refinedGrammar && (
          <div className="grammar-correction fade-in">
            <CheckCircle2 size={16} className="grammar-icon" />
            <div className="grammar-text">
              <strong>Context Detected: </strong>
              {refinedGrammar}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
