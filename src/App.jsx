import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Volume2, VolumeX, WifiOff, Sparkles, CheckCircle2 } from 'lucide-react';
import './App.css';

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);

  const [confidence, setConfidence] = useState(0);

  const [fullSentence, setFullSentence] = useState([]);
  const [lastDetectedWord, setLastDetectedWord] = useState('');
  const [stabilityCount, setStabilityCount] = useState(0);
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
  const lastVideoTimeRef = useRef(-1);
  const stabilityThreshold = 15; // Number of frames to stay on the same word before adding it

  // Initialize Core Edge ML Models Unblocked
  useEffect(() => {
    const loadModels = async () => {
      try {
        await tf.ready();

        // Reconstruct the model manually and load weights
        try {
          const resLabels = await fetch('/models/isl_model/labels.json');
          labelsRef.current = await resLabels.json();

          const resWeights = await fetch('/models/isl_model/weights.json');
          const weightsData = await resWeights.json();

          const NUM_CLASSES = labelsRef.current.length;
          const NUM_COORD_POINTS = 126;

          const model = tf.sequential();
          model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [NUM_COORD_POINTS] }));
          model.add(tf.layers.dropout({ rate: 0.2 }));
          model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
          model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

          // Set weights
          let weightIdx = 0;
          model.layers.forEach(layer => {
            if (layer.getWeights().length > 0) {
              const layerWeights = weightsData[weightIdx];
              layer.setWeights([
                tf.tensor(layerWeights.weights),
                tf.tensor(layerWeights.biases)
              ]);
              weightIdx++;
            }
          });

          customAiModelRef.current = model;
          setModelLoaded(true);
          console.log("Custom Neural Network loaded successfully via Weights Sync.");
        } catch (e) {
          console.error("AI Model Reconstruction Error:", e);
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

  const predictLoop = useCallback(async function loop() {
    if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current || !customAiModelRef.current) return;

    const video = videoRef.current;
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

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

          if (confidenceScore > 0.6) { // Increased threshold for stability
            const detectedLabel = labelsRef.current[maxIdx] || 'Recognizing...';
            setCurrentSentence(detectedLabel);
            setConfidence(Math.round(confidenceScore * 100));

            // Sentence building logic
            if (detectedLabel === lastDetectedWord) {
              setStabilityCount(prev => {
                const newCount = prev + 1;
                if (newCount === stabilityThreshold) {
                  setFullSentence(prevSentence => {
                    // Avoid adding the same word twice in a row immediately
                    if (prevSentence.length > 0 && prevSentence[prevSentence.length - 1] === detectedLabel) {
                      return prevSentence;
                    }
                    return [...prevSentence, detectedLabel];
                  });
                }
                return newCount;
              });
            } else {
              setLastDetectedWord(detectedLabel);
              setStabilityCount(0);
            }

            setRefinedGrammar(`Recognized Word: "${detectedLabel}" (Stability: ${Math.round((stabilityCount / stabilityThreshold) * 100)}%)`);
          } else {
            setConfidence(0);
            setCurrentSentence('');
            setStabilityCount(0);
          }
          tensor.dispose();
        } else {
          setIsProcessing(false);
          setConfidence(0);
          setStabilityCount(0);
        }
      }
    }

    if (isCameraActive) {
      requestRef.current = requestAnimationFrame(loop);
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
          {!isCameraActive && fullSentence.length === 0 ? (
            <div style={{ opacity: 0.3 }}>
              <div className="skeleton-text"></div>
              <div className="skeleton-text" style={{ width: '60%' }}></div>
            </div>
          ) : (
            <div className="live-text">
              {!modelLoaded ?
                "Waiting for Custom Neural Network..." :
                (fullSentence.length > 0 ? fullSentence.join(' ') : (currentSentence || 'Waiting for gestures...'))}
              {isProcessing && <span className="cursor" />}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
          <button
            className="glass-panel"
            style={{ padding: '8px 16px', cursor: 'pointer', border: 'none', color: 'white', backgroundColor: 'rgba(255,50,50,0.3)' }}
            onClick={() => setFullSentence([])}
          >
            Clear Sentence
          </button>
          <button
            className="glass-panel"
            style={{ padding: '8px 16px', cursor: 'pointer', border: 'none', color: 'white', backgroundColor: 'rgba(50,255,100,0.2)' }}
            onClick={() => setFullSentence(prev => [...prev, ' '])}
          >
            Add Space
          </button>
        </div>

        {refinedGrammar && (
          <div className="grammar-correction fade-in" style={{ marginTop: '15px' }}>
            <CheckCircle2 size={16} className="grammar-icon" />
            <div className="grammar-text">
              <strong>Status: </strong>
              {refinedGrammar}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
