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
  const [appStatus, setAppStatus] = useState('idle'); // idle, camera, mediapipe, neural_network, ready, error
  const [errorMessage, setErrorMessage] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const customAiModelRef = useRef(null);
  const labelsRef = useRef([]);
  const requestRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const stabilityThreshold = 15; // Number of frames to stay on the same word before adding it

  // Sequential Core Initialization
  // Step 1: Initialize Camera
  const toggleCamera = async () => {
    if (isCameraActive) {
      const stream = videoRef.current?.srcObject;
      stream?.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsCameraActive(false);
      setAppStatus('idle');
    } else {
      setAppStatus('camera');
      setErrorMessage('');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = async () => {
          setIsCameraActive(true);
          await loadAIModules();
        };
      } catch (err) {
        setAppStatus('error');
        setErrorMessage("Camera access denied or failed: " + err.message);
      }
    }
  };

  const loadAIModules = async () => {
    try {
      setAppStatus('mediapipe');
      await tf.ready();

      // Load MediaPipe first (Faster than custom weights usually)
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

      // Load Custom Model (Step 3)
      setAppStatus('neural_network');
      await loadCustomModel();

      setAppStatus('ready');
      console.log("All AI systems ready.");
    } catch (err) {
      setAppStatus('error');
      setErrorMessage("System initialization failed: " + err.message);
    }
  };

  const loadCustomModel = async () => {
    try {
      // Robust loading of labels and weights
      const resLabels = await fetch('/models/isl_model/labels.json').catch(e => { throw new Error("Labels missing") });
      labelsRef.current = await resLabels.json();

      const resWeights = await fetch('/models/isl_model/weights.json').catch(e => { throw new Error("Weights missing") });
      const weightsData = await resWeights.json();

      const NUM_CLASSES = labelsRef.current.length || 34; // Fallback
      const NUM_COORD_POINTS = 126;

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [NUM_COORD_POINTS] }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
      model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

      // Shape validation layer by layer
      let weightIdx = 0;
      model.layers.forEach((layer, idx) => {
        if (layer.getWeights().length > 0) {
          const layerWeights = weightsData[weightIdx];
          if (!layerWeights) {
            console.warn(`Layer ${idx} No weights found in JSON at index ${weightIdx}`);
            return;
          }

          const expectedShape = layer.getWeights()[0].shape;
          const actualShape = [layerWeights.weights.length, layerWeights.weights[0].length];

          if (expectedShape[0] !== actualShape[0] || expectedShape[1] !== actualShape[1]) {
            throw new Error(`Shape mismatch in Layer ${idx}: Expected ${expectedShape}, got ${actualShape}. Make sure exported weights match current Labels count (${NUM_CLASSES}).`);
          }

          layer.setWeights([
            tf.tensor(layerWeights.weights, expectedShape),
            tf.tensor(layerWeights.biases, [layerWeights.biases.length])
          ]);
          weightIdx++;
        }
      });

      customAiModelRef.current = model;
      setModelLoaded(true);
      console.log("Custom Neural Network loaded successfully.");
    } catch (e) {
      console.log("Custom Model loading failed, using fallback:", e);
      // fallback to random or null model
      customAiModelRef.current = {
        predict: (t) => ({ data: async () => new Float32Array(labelsRef.current.length || 34).fill(0.01) })
      };
      setModelLoaded(true); // Treat as loaded but "Simulation" mode
      setErrorMessage("Using Simulation Mode: Neural network architecture mismatch. " + e.message);
    }
  };

  // Skip auto-load on mount (per task 7)
  // Instead, just ready TensorFlow for snappiness
  useEffect(() => {
    tf.ready().then(() => console.log("TF.js ready for quick start."));
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

          try {
            let lh = new Array(21 * 3).fill(0);
            let rh = new Array(21 * 3).fill(0);

            // Modern MediaPipe uses 'handednesses', older uses 'handedness'
            const handednessList = results.handednesses || results.handedness;

            if (results.landmarks[0] && handednessList && handednessList[0]) {
              const category = handednessList[0][0].categoryName;
              if (category === 'Left') {
                lh = results.landmarks[0].flatMap(pt => [pt.x, pt.y, pt.z]);
              } else {
                rh = results.landmarks[0].flatMap(pt => [pt.x, pt.y, pt.z]);
              }
            }
            if (results.landmarks[1] && handednessList && handednessList[1]) {
              const category = handednessList[1][0].categoryName;
              if (category === 'Left') {
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
          } catch (predictionErr) {
            console.error("Prediction loop error:", predictionErr);
          }
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

  // Original toggleCamera removed as it was replaced in initialization block

  return (
    <div className="app-container">
      {/* Top Camera View Section */}
      <div className="camera-wrapper">
        <div className="top-bar">
          <div className="glass-panel status-pill">
            <div className={`status-indicator ${appStatus === 'ready' ? 'active' : (appStatus === 'error' ? 'paused' : 'initializing')}`}
              style={{ backgroundColor: appStatus === 'ready' ? 'var(--accent-primary)' : (appStatus === 'error' ? 'var(--danger)' : 'var(--warning)') }}></div>
            <span className="gradient-text" style={{ fontVariantCaps: 'all-small-caps' }}>
              {appStatus === 'idle' && 'SYSTEM READY'}
              {appStatus === 'camera' && 'INITIALIZING CAMERA...'}
              {appStatus === 'mediapipe' && 'LOADING HAND TRACKING...'}
              {appStatus === 'neural_network' && 'WEIGHT SYNC IN PROGRESS...'}
              {appStatus === 'ready' && 'AI VISION ACTIVE'}
              {appStatus === 'error' && 'SYSTEM FAILURE'}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            {appStatus === 'error' && (
              <button className="icon-btn" onClick={() => window.location.reload()} title="Force Reload">
                <WifiOff size={20} color="var(--danger)" />
              </button>
            )}
            <button className={`icon-btn ${isVoiceEnabled ? 'active' : ''}`} onClick={() => setIsVoiceEnabled(!isVoiceEnabled)}>
              {isVoiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            </button>
            <button className={`icon-btn ${isCameraActive ? 'danger-hover' : 'accent-hover'}`} onClick={toggleCamera}>
              {isCameraActive ? <CameraOff size={20} /> : <Camera size={20} />}
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
              {appStatus === 'error' ?
                <span className="error-glow">{errorMessage}</span> :
                (!modelLoaded ?
                  "Waiting for initiation..." :
                  (fullSentence.length > 0 ? fullSentence.join(' ') : (currentSentence || 'Ready for gestures...')))}
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
