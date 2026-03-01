import { useState, useRef, useEffect } from 'react';
import { Camera, CameraOff, Volume2, VolumeX, WifiOff, RefreshCcw, Settings, Sparkles, CheckCircle2 } from 'lucide-react';
import './App.css';

function App() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [isOffline, setIsOffline] = useState(false);
  const [confidence, setConfidence] = useState(0);
  
  const videoRef = useRef(null);
  
  // Simulated tracking state
  const [currentSentence, setCurrentSentence] = useState('');
  const [refinedGrammar, setRefinedGrammar] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Simulated AI Engine Flow
  useEffect(() => {
    if (!isCameraActive) {
      setCurrentSentence('');
      setRefinedGrammar('');
      setConfidence(0);
      return;
    }

    const phrases = ["Hello", "Hello how", "Hello how are", "Hello how are you"];
    const correctedPhrases = ["", "", "", "Hello, how are you doing today?"];
    let step = 0;

    const interval = setInterval(() => {
      if (step < phrases.length) {
        setIsProcessing(true);
        setCurrentSentence(phrases[step]);
        setConfidence(70 + Math.random() * 25);
        if (correctedPhrases[step]) {
          setRefinedGrammar(correctedPhrases[step]);
        }
        step++;
      } else {
        setIsProcessing(false);
        setConfidence(98);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [isCameraActive]);

  const toggleCamera = async () => {
    if (isCameraActive) {
      const stream = videoRef.current?.srcObject;
      const tracks = stream?.getTracks();
      tracks?.forEach(track => track.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsCameraActive(true);
      } catch (err) {
        console.error("Error accessing camera:", err);
        // Fallback or mock if camera fails
        setIsCameraActive(true);
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
              {isCameraActive ? 'AI VISION ACTIVE' : 'CAMERA OFF'}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            {isOffline && (
              <button className="icon-btn active" title="Offline Mode Supported">
                <WifiOff size={20} />
              </button>
            )}
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
        
        {!isCameraActive && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            <CameraOff size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <p style={{ fontFamily: 'Outfit, sans-serif' }}>Tap camera icon to begin translating</p>
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
              {currentSentence || 'Waiting for gestures...'}
              {isProcessing && <span className="cursor" />}
            </div>
          )}
        </div>

        {/* AI Sentence Refinement (Context / Grammar) */}
        {refinedGrammar && (
          <div className="grammar-correction fade-in">
            <CheckCircle2 size={16} className="grammar-icon" />
            <div className="grammar-text">
              <strong>Context Refined: </strong>
              {refinedGrammar}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
