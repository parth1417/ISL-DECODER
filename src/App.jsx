import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Volume2, VolumeX, WifiOff, Sparkles, Delete, Mic } from 'lucide-react';
import './App.css';

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

// How many consecutive frames the SAME letter must be detected before it "locks in"
const STABILITY_THRESHOLD = 30; // ~1 second at 30fps

function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);

  const [confidence, setConfidence] = useState(0);
  const [currentLetter, setCurrentLetter] = useState('');    // Live detected letter this frame
  const [stabilityProgress, setStabilityProgress] = useState(0); // 0–100% progress to lock-in
  const [builtSentence, setBuiltSentence] = useState('');    // The accumulated sentence string
  const [modelLoaded, setModelLoaded] = useState(false);
  const [appStatus, setAppStatus] = useState('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [isHandVisible, setIsHandVisible] = useState(false);
  const [debugLogs, setDebugLogs] = useState([]);

  const addLog = (msg) => {
    console.log(`[AI-LOG] ${msg}`);
    setDebugLogs(prev => [`${new Date().toLocaleTimeString([], { hour12: false })}: ${msg}`, ...prev].slice(0, 8));
  };

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const customAiModelRef = useRef(null);
  const labelsRef = useRef([]);
  const requestRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  // Refs for prediction loop (avoid stale closure issues)
  const stabilityCountRef = useRef(0);
  const lastDetectedLetterRef = useRef('');
  const builtSentenceRef = useRef('');

  // Keep builtSentenceRef in sync with state
  useEffect(() => { builtSentenceRef.current = builtSentence; }, [builtSentence]);

  // ─── TTS helper ────────────────────────────────────────────────────────────
  const speak = useCallback((text) => {
    if (!isVoiceEnabled || !text.trim()) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.trim());
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  }, [isVoiceEnabled]);

  // ─── Camera & AI Init ──────────────────────────────────────────────────────
  const toggleCamera = async () => {
    console.log("Toggle Camera: ", !isCameraActive);
    if (isCameraActive) {
      const stream = videoRef.current?.srcObject;
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
      if (videoRef.current) videoRef.current.srcObject = null;
      setIsCameraActive(false);
      setAppStatus('idle');
      setCurrentLetter('');
      setStabilityProgress(0);
      setIsHandVisible(false);
    } else {
      setAppStatus('camera');
      setErrorMessage('');
      try {
        addLog("Requesting standard camera...");
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: true
        });
        
        if (videoRef.current) {
          videoRef.current.onloadedmetadata = async () => {
             console.log("Video metadata loaded");
             try {
               await videoRef.current.play();
               console.log("Video playing");
               await loadAIModules();
             } catch (playErr) {
               console.error("Play error:", playErr);
               // Even if play fails (e.g. autonomy block), try to continue
               await loadAIModules();
             }
          };
          
          videoRef.current.srcObject = stream;
          setIsCameraActive(true);
        }
      } catch (err) {
        console.error("Camera Access Error:", err);
        setAppStatus('error');
        const msg = err.name === 'NotAllowedError' ? 'Permission Denied. Please allow camera in settings.' : err.message;
        setErrorMessage(`Camera Error: ${msg}`);
      }
    }
  };

  const loadAIModules = async () => {
    if (handLandmarkerRef.current) return;
    try {
      addLog("Booting AI Engine (v1.5)...");
      setAppStatus('mediapipe');
      await tf.ready();
      
      addLog("Loading MediaPipe Runtime...");
      // Using 0.10.14 as it is highly stable via CDN
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
      );
      
      addLog("Configuring Hand Tracker...");
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/lite/1/hand_landmarker.task',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.3,
      });
      
      addLog("Syncing Custom Model...");
      setAppStatus('neural_network');
      await loadCustomModel();
      
      setAppStatus('ready');
      setErrorMessage('');
      addLog("SYSTEM ONLINE.");
    } catch (err) {
      const detail = `${err.name}: ${err.message}`;
      console.error(err);
      addLog(`FATAL: ${detail}`);
      setAppStatus('error');
      setErrorMessage(detail);
      // Popup alert for immediate mobile debugging
      alert(`DIAGNOSTIC v1.5: ${detail}`);
    }
  };

  const loadCustomModel = async () => {
    try {
      addLog("Loading Model Assets...");
      const resLabels = await fetch('/models/isl_model/labels.json');
      if (!resLabels.ok) throw new Error(`Labels Fetch 404`);
      labelsRef.current = await resLabels.json();

      const resWeights = await fetch('/models/isl_model/weights.json');
      if (!resWeights.ok) throw new Error(`Weights Fetch 404`);
      const weightsData = await resWeights.json();

      const numLabels = labelsRef.current.length;
      addLog(`Syncing ${numLabels} classes...`);
      
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [126] }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
      model.add(tf.layers.dense({ units: numLabels, activation: 'softmax' }));

      let weightIdx = 0;
      for (let layer of model.layers) {
        if (layer.getWeights().length > 0) {
          const lw = weightsData[weightIdx];
          if (!lw) continue;
          layer.setWeights([
            tf.tensor(lw.weights, layer.getWeights()[0].shape),
            tf.tensor(lw.biases, [lw.biases.length]),
          ]);
          weightIdx++;
        }
      }

      customAiModelRef.current = model;
      setModelLoaded(true);
      addLog("ISL Classifier: Ready.");
    } catch (e) {
      addLog(`Classifier Error: ${e.message}`);
      customAiModelRef.current = {
        predict: (t) => ({ data: async () => new Float32Array(labelsRef.current?.length || 34).fill(0.0) }),
      };
      setModelLoaded(true);
      setErrorMessage(`Model: ${e.message}`);
    }
  };

  useEffect(() => {
    tf.ready().then(() => console.log('TF.js ready.'));
  }, []);

  // ─── Landmark Drawing ──────────────────────────────────────────────────────
  const drawLandmarks = (ctx, landmarks) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // MediaPipe Hand Connections (Standard Skeleton)
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [5, 9], [9, 10], [10, 11], [11, 12], // Middle
      [9, 13], [13, 14], [14, 15], [15, 16], // Ring
      [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
    ];

    for (const hand of landmarks) {
      // Draw Connections (Lines)
      ctx.strokeStyle = '#00b8ff';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      
      for (const [start, end] of connections) {
        if (hand[start] && hand[end]) {
          ctx.beginPath();
          ctx.moveTo(hand[start].x * ctx.canvas.width, hand[start].y * ctx.canvas.height);
          ctx.lineTo(hand[end].x * ctx.canvas.width, hand[end].y * ctx.canvas.height);
          ctx.stroke();
        }
      }

      // Draw Landmarks (Points)
      ctx.fillStyle = '#00ff88';
      for (const point of hand) {
        ctx.beginPath();
        ctx.arc(point.x * ctx.canvas.width, point.y * ctx.canvas.height, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  };

  // ─── Prediction Loop ───────────────────────────────────────────────────────
  const predictLoop = useCallback(async function loop() {
    if (!videoRef.current || !canvasRef.current || !handLandmarkerRef.current || !customAiModelRef.current) return;

    const video = videoRef.current;
    if (video.readyState < 2) return; // Wait for HAVE_CURRENT_DATA

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      // Use a timestamp that matches the video metadata if possible
      const results = handLandmarkerRef.current.detectForVideo(video, performance.now());
      const ctx = canvasRef.current.getContext('2d');
      
      // Only set dimensions once or when they change to avoid unnecessary clears
      if (canvasRef.current.width !== video.videoWidth || canvasRef.current.height !== video.videoHeight) {
        canvasRef.current.width = video.videoWidth || 640;
        canvasRef.current.height = video.videoHeight || 480;
      }

      if (results.landmarks && results.landmarks.length > 0) {
        drawLandmarks(ctx, results.landmarks);
        setIsHandVisible(true);
        if (Math.random() < 0.05) console.log("Hand Landmarks detected:", results.landmarks.length);

        try {
          let lh = new Array(21 * 3).fill(0);
          let rh = new Array(21 * 3).fill(0);
          const handednessList = results.handednesses || results.handedness;

          const normalizePts = (pts) => {
            const wrist = pts[0];
            const translated = pts.map(pt => ({
                x: pt.x - wrist.x,
                y: pt.y - wrist.y,
                z: pt.z - wrist.z
            }));
            let maxVal = 0;
            for (const pt of translated) {
              maxVal = Math.max(maxVal, Math.abs(pt.x), Math.abs(pt.y), Math.abs(pt.z));
            }
            return translated.flatMap(pt => [
                maxVal > 0 ? pt.x / maxVal : 0,
                maxVal > 0 ? pt.y / maxVal : 0,
                maxVal > 0 ? pt.z / maxVal : 0
            ]);
          };

          if (results.landmarks[0] && handednessList?.[0]) {
            const cat = handednessList[0][0].categoryName;
            if (cat === 'Left') lh = normalizePts(results.landmarks[0]);
            else rh = normalizePts(results.landmarks[0]);
          }
          if (results.landmarks[1] && handednessList?.[1]) {
            const cat = handednessList[1][0].categoryName;
            if (cat === 'Left') lh = normalizePts(results.landmarks[1]);
            else rh = normalizePts(results.landmarks[1]);
          }

          const tensor = tf.tensor2d([lh.concat(rh)]);
          const prediction = await customAiModelRef.current.predict(tensor).data();
          const maxIdx = prediction.indexOf(Math.max(...prediction));
          const confidenceScore = prediction[maxIdx];
          tensor.dispose();

          if (confidenceScore > 0.6) {
            const detected = labelsRef.current[maxIdx] || '?';
            setCurrentLetter(detected);
            setConfidence(Math.round(confidenceScore * 100));

            // Stability check — same letter for STABILITY_THRESHOLD frames → lock it in
            if (detected === lastDetectedLetterRef.current) {
              stabilityCountRef.current += 1;
              const progress = Math.min(100, Math.round((stabilityCountRef.current / STABILITY_THRESHOLD) * 100));
              setStabilityProgress(progress);

              if (stabilityCountRef.current === STABILITY_THRESHOLD) {
                // Lock this letter into the sentence
                setBuiltSentence(prev => prev + detected);
                builtSentenceRef.current = builtSentenceRef.current + detected;
                stabilityCountRef.current = 0; // reset so next hold adds another
              }
            } else {
              lastDetectedLetterRef.current = detected;
              stabilityCountRef.current = 0;
              setStabilityProgress(0);
            }
          } else {
            setCurrentLetter('');
            setConfidence(0);
            setStabilityProgress(0);
            stabilityCountRef.current = 0;
          }
        } catch (err) {
          console.error('Prediction error:', err);
        }
      } else {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        setIsHandVisible(false);
        setCurrentLetter('');
        setConfidence(0);
        setStabilityProgress(0);
        stabilityCountRef.current = 0;
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

  // ─── Sentence Actions ──────────────────────────────────────────────────────
  const addSpace = () => {
    const word = builtSentence.split(' ').pop(); // last word
    if (word) speak(word);
    setBuiltSentence(prev => prev + ' ');
  };

  const backspace = () => {
    setBuiltSentence(prev => prev.slice(0, -1));
  };

  const clearSentence = () => {
    setBuiltSentence('');
    builtSentenceRef.current = '';
  };

  const speakSentence = () => speak(builtSentence);

  // ─── UI ────────────────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      {/* Camera Section */}
      <div className="camera-wrapper">
        <div className="top-bar">
          <div className="glass-panel status-pill">
            <div
              className={`status-indicator ${appStatus === 'ready' ? 'active' : appStatus === 'error' ? 'paused' : 'initializing'}`}
              style={{
                backgroundColor:
                  appStatus === 'ready' ? 'var(--accent-primary)' :
                    appStatus === 'error' ? 'var(--danger)' : 'var(--warning)',
              }}
            />
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
            <button className="icon-btn" onClick={loadAIModules} title="Debug Init">
              <Sparkles size={20} />
            </button>
            <button className={`icon-btn ${isCameraActive ? 'danger-hover' : 'accent-hover'}`} onClick={toggleCamera}>
              {isCameraActive ? <CameraOff size={20} /> : <Camera size={20} />}
            </button>
          </div>
        </div>

        <video ref={videoRef} autoPlay playsInline muted className="camera-feed" style={{ display: isCameraActive ? 'block' : 'none' }} />
        <canvas ref={canvasRef} className="tracking-canvas" style={{ display: isCameraActive ? 'block' : 'none' }} />

        {/* Live Letter HUD — shown while camera is active */}
        {isCameraActive && (
          <div className="live-letter-hud">
            <div className="live-letter-display">
              {currentLetter || (isHandVisible ? '?' : '✋')}
            </div>
            <div className="stability-bar-wrapper">
              <div
                className="stability-bar-fill"
                style={{ width: `${stabilityProgress}%`, backgroundColor: stabilityProgress === 100 ? 'var(--accent-primary)' : 'var(--warning)' }}
              />
            </div>
            <div className="live-letter-label">
              {currentLetter
                ? `Detecting: "${currentLetter}" — hold still to add  (${stabilityProgress}%)`
                : isHandVisible ? 'Low confidence...' : 'Show your hand'}
            </div>
            {confidence > 0 && (
              <div className="confidence-chip">{confidence}% confident</div>
            )}
          </div>
        )}

        {!isCameraActive && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            <CameraOff size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <p style={{ fontFamily: 'Outfit, sans-serif' }}>Tap camera icon to begin AI tracking</p>
            {appStatus === 'error' && (
               <button 
                 onClick={() => { setAppStatus('idle'); setIsCameraActive(true); toggleCamera(); }}
                 style={{ marginTop: '20px', background: 'var(--danger)', color: 'white', padding: '10px 20px', borderRadius: '8px' }}
               >
                 Fix & Restart AI
               </button>
            )}
          </div>
        )}
      </div>

      {/* Output Panel */}
      <div className="output-panel">
        <div className="panel-header" style={{ marginBottom: '8px' }}>
          <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={18} className="gradient-text" />
            Sentence Builder
          </h2>
          {debugLogs.length > 0 && (
            <div style={{ fontSize: '0.6rem', color: 'var(--accent-primary)', opacity: 0.8 }}>
              v1.5 - {debugLogs[0]}
            </div>
          )}
        </div>

        {/* Diagnostic Logs - Only show on failure or for dev */}
        {appStatus === 'error' && (
          <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '8px', border: '1px solid #333', marginBottom: '10px' }}>
            <div style={{ fontSize: '0.7rem', color: '#666', marginBottom: '5px', textTransform: 'uppercase' }}>Kernel Diagnostics</div>
            {debugLogs.map((log, i) => (
              <div key={i} style={{ fontSize: '0.75rem', color: i === 0 ? '#00ff88' : '#aaa', fontFamily: 'monospace' }}>{log}</div>
            ))}
          </div>
        )}

        {/* How-to hint */}
        <div className="hint-strip">
          ✋ Hold a sign steady for ~1 sec to add a letter &nbsp;·&nbsp; Use <strong>Space</strong> to separate words &nbsp;·&nbsp; <strong>Backspace</strong> to fix mistakes
        </div>

        <div className="sentence-display" style={{ minHeight: '120px', flex: 'none', background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '15px' }}>
          {appStatus === 'error' ? (
            <div style={{ color: '#ff4a4a', textAlign: 'center' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>⚠️ SYSTEM FAILURE</div>
              <div style={{ fontSize: '0.8rem', marginBottom: '12px', background: 'rgba(255,0,0,0.1)', padding: '5px', borderRadius: '4px' }}>
                {errorMessage}
              </div>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button 
                  onClick={() => window.location.reload()}
                  style={{ background: '#333', padding: '8px 15px', borderRadius: '6px', fontSize: '0.8rem' }}
                >
                  Reload Page
                </button>
                <button 
                  onClick={() => { setAppStatus('idle'); toggleCamera(); }}
                  style={{ background: '#ff4a4a', color: 'white', padding: '8px 15px', borderRadius: '6px', fontSize: '0.8rem' }}
                >
                  Try Again
                </button>
              </div>
            </div>
          ) : builtSentence ? (
            <>
              <span className="sentence-text">{builtSentence}</span>
              <span className="cursor" />
            </>
          ) : (
            <span style={{ opacity: 0.3, fontStyle: 'italic' }}>
              {modelLoaded ? 'Start signing to build a sentence...' : 'Waiting for model...'}
            </span>
          )}
        </div>

        {/* Action buttons */}
        <div className="action-buttons">
          <button className="action-btn space-btn" onClick={addSpace} title="Add space between words">
            ␣ Space
          </button>
          <button className="action-btn backspace-btn" onClick={backspace} title="Delete last character">
            <Delete size={16} /> Backspace
          </button>
          <button className="action-btn speak-btn" onClick={speakSentence} title="Read sentence aloud" disabled={!builtSentence.trim()}>
            <Mic size={16} /> Speak
          </button>
          <button className="action-btn clear-btn" onClick={clearSentence} title="Clear the sentence">
            ✕ Clear
          </button>
        </div>

        {/* Error hint */}
        {errorMessage && appStatus !== 'error' && (
          <div className="grammar-correction fade-in" style={{ marginTop: '10px' }}>
            <span style={{ opacity: 0.6, fontSize: '0.8rem' }}>⚠ {errorMessage}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
