// offscreen.js — Offscreen document
// Owns tab audio capture and the persistent backend WebSocket session.

let audioContext;
let mediaStream;
let source;
let workletNode;
let backendSocket;
let currentSessionId = null;

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'START_CAPTURE' && message.offscreen) {
    startCapture(message.streamId);
  } else if (message.type === 'STOP_CAPTURE' && message.offscreen) {
    stopCapture();
  } else if (message.type === 'REQUEST_SUMMARY' && message.offscreen) {
    requestSummary(message.requestId);
  }
});

async function startCapture(streamId) {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      },
      video: false
    });

    audioContext = new AudioContext({
      sampleRate: 16000
    });

    source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(audioContext.destination);

    await openBackendConnection();
    await setupWorklet();
  } catch (err) {
    chrome.runtime.sendMessage({
      type: 'API_ERROR',
      source: 'Offscreen',
      message: `Failed to start audio capture: ${err.message}`
    }).catch(() => {});
  }
}

async function stopCapture() {
  if (backendSocket) {
    try {
      if (backendSocket.readyState === WebSocket.OPEN) {
        backendSocket.send(JSON.stringify({ type: 'stop_session' }));
      }
    } catch (err) {
    }
    backendSocket.close();
    backendSocket = null;
  }

  currentSessionId = null;

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }

  if (source) {
    source.disconnect();
    source = null;
  }

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

async function setupWorklet() {
  await audioContext.audioWorklet.addModule('capture-worklet.js');
  workletNode = new AudioWorkletNode(audioContext, 'capture-worklet');

  workletNode.port.onmessage = (event) => {
    const pcmBuffer = event.data;
    if (backendSocket && backendSocket.readyState === WebSocket.OPEN) {
      backendSocket.send(pcmBuffer);
    }
  };

  source.connect(workletNode);
}

function openBackendConnection() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(CONFIG.BACKEND_WS_URL);
    socket.binaryType = 'arraybuffer';

    let opened = false;

    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: 'start_session',
        config: {
          deepgramParams: CONFIG.DEEPGRAM_PARAMS || {},
          geminiModel: CONFIG.GEMINI_MODEL || null
        }
      }));
      opened = true;
      resolve();
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleBackendMessage(message);
      } catch (err) {
        chrome.runtime.sendMessage({
          type: 'API_ERROR',
          source: 'Backend',
          message: `Invalid backend message: ${err.message}`
        }).catch(() => {});
      }
    };

    socket.onerror = () => {
      if (!opened) {
        reject(new Error('Failed to connect to backend WebSocket'));
      }
      chrome.runtime.sendMessage({
        type: 'API_ERROR',
        source: 'Backend',
        message: 'A backend WebSocket error occurred. Verify the FastAPI server is running.'
      }).catch(() => {});
    };

    socket.onclose = (event) => {
      if (backendSocket === socket) {
        backendSocket = null;
      }

      if (event.code !== 1000 && event.code !== 1005) {
        chrome.runtime.sendMessage({
          type: 'API_ERROR',
          source: 'Backend',
          message: `Backend WebSocket closed unexpectedly (${event.code}).`
        }).catch(() => {});
      }
    };

    backendSocket = socket;
  });
}

function handleBackendMessage(message) {
  if (message.type === 'session_started') {
    currentSessionId = message.sessionId || null;
    chrome.runtime.sendMessage({
      type: 'SESSION_READY',
      sessionId: currentSessionId
    }).catch(() => {});
    return;
  }

  if (message.type === 'transcript_update') {
    chrome.runtime.sendMessage({
      type: 'TRANSCRIPT_RECEIVED',
      transcript: message.transcript,
      isFinal: message.isFinal,
      metadata: message.metadata
    }).catch(() => {});
    return;
  }

  if (message.type === 'utterance_end') {
    chrome.runtime.sendMessage({ type: 'UTTERANCE_END' }).catch(() => {});
    return;
  }

  if (message.type === 'utterance_committed') {
    chrome.runtime.sendMessage({
      type: 'UTTERANCE_COMMITTED',
      utteranceId: message.utteranceId,
      text: message.text
    }).catch(() => {});
    return;
  }

  if (message.type === 'ai_response_chunk') {
    chrome.runtime.sendMessage({
      type: 'AI_RESPONSE_CHUNK',
      utteranceId: message.utteranceId,
      text: message.text
    }).catch(() => {});
    return;
  }

  if (message.type === 'ai_response_done') {
    // Send standard DONE event
    chrome.runtime.sendMessage({
      type: 'AI_RESPONSE_DONE',
      utteranceId: message.utteranceId,
      fullText: message.fullText,
      badgeType: message.badgeType
    }).catch(() => {});

    // Also send a special CHUNK event with isDone=true to signal sidepanel
    // that it should replace its interim text with the finalized normalized text
    chrome.runtime.sendMessage({
      type: 'AI_RESPONSE_CHUNK',
      utteranceId: message.utteranceId,
      text: '',
      isDone: true,
      finalText: message.fullText
    }).catch(() => {});
    return;
  }

  if (message.type === 'error') {
    chrome.runtime.sendMessage({
      type: 'API_ERROR',
      source: message.source || 'Backend',
      message: message.message || 'Unknown backend error'
    }).catch(() => {});
  }
}

async function requestSummary(requestId) {
  let summary;

  if (!currentSessionId) {
    summary = {
      extractedData: {},
      insights: ['No backend session is available for summary generation.']
    };
  } else {
    try {
      const response = await fetch(`${CONFIG.BACKEND_HTTP_URL}/api/sessions/${currentSessionId}/summary`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      summary = await response.json();
    } catch (err) {
      summary = {
        extractedData: {},
        insights: [`Failed to fetch summary from backend: ${err.message}`]
      };
    }
  }

  chrome.runtime.sendMessage({
    type: 'SUMMARY_RESULT',
    requestId,
    summary
  }).catch(() => {});
}
