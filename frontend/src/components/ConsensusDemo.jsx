import React, { useState } from 'react';
import './ConsensusDemo.css';

const ConsensusDemo = () => {
  const [message, setMessage] = useState('');
  const [sessionId] = useState('demo-session-' + Date.now());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const exampleMessages = [
    { label: '🚨 Crisis', text: 'I want to kill myself', category: 'crisis' },
    { label: '🚨 Crisis with Plan', text: "I'm going to end my life tonight", category: 'crisis' },
    { label: '⚠️ Caution', text: 'I feel hopeless and worthless', category: 'caution' },
    { label: '⚠️ Depression', text: "I can't sleep and nothing matters anymore", category: 'caution' },
    { label: '✅ Hyperbole', text: 'This homework is killing me', category: 'safe' },
    { label: '✅ Stress', text: "I'm stressed about exams", category: 'safe' },
  ];

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      // Fetch the detailed conversation to get scores
      const convResponse = await fetch(`http://localhost:8000/conversations/${sessionId}`);
      const conversations = await convResponse.json();
      const latestConv = conversations[0]; // Most recent

      setResult({
        message: text,
        response: data.response,
        riskLevel: data.risk_level,
        isCrisis: data.is_crisis,
        conversationId: data.conversation_id,
        details: latestConv,
      });

      setHistory([...history, {
        message: text,
        riskLevel: data.risk_level,
        isCrisis: data.is_crisis,
        timestamp: new Date().toLocaleTimeString(),
      }]);

      setMessage('');
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to send message. Make sure the backend is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'CRISIS': return '#dc3545';
      case 'CAUTION': return '#ffc107';
      case 'SAFE': return '#28a745';
      default: return '#6c757d';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'CRISIS': return '🚨';
      case 'CAUTION': return '⚠️';
      case 'SAFE': return '✅';
      default: return '❓';
    }
  };

  return (
    <div className="consensus-demo">
      <div className="demo-header">
        <h1>🧠 PsyFlo Consensus Demo</h1>
        <p>Watch the 3-way consensus in action: Regex + Semantic + Mistral</p>
      </div>

      <div className="demo-container">
        {/* Left Panel - Input & Examples */}
        <div className="input-panel">
          <div className="message-input-section">
            <h3>Test Message</h3>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type a message to analyze..."
              rows="4"
              disabled={loading}
            />
            <button
              onClick={() => sendMessage(message)}
              disabled={loading || !message.trim()}
              className="send-button"
            >
              {loading ? 'Analyzing...' : 'Analyze Message'}
            </button>
          </div>

          <div className="examples-section">
            <h3>Quick Examples</h3>
            <div className="example-buttons">
              {exampleMessages.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => sendMessage(example.text)}
                  disabled={loading}
                  className={`example-button ${example.category}`}
                >
                  {example.label}
                  <span className="example-text">{example.text}</span>
                </button>
              ))}
            </div>
          </div>

          {/* History */}
          {history.length > 0 && (
            <div className="history-section">
              <h3>Analysis History</h3>
              <div className="history-list">
                {history.map((item, idx) => (
                  <div key={idx} className="history-item">
                    <span className="history-icon">{getRiskIcon(item.riskLevel)}</span>
                    <span className="history-message">{item.message.substring(0, 40)}...</span>
                    <span className="history-time">{item.timestamp}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="results-panel">
          {!result && !loading && (
            <div className="empty-state">
              <h2>👈 Send a message to see the consensus in action</h2>
              <p>The system will analyze it using three strategies and show you how they vote.</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Analyzing with 3-way consensus...</p>
            </div>
          )}

          {result && (
            <div className="result-display">
              {/* Message Being Analyzed */}
              <div className="analyzed-message">
                <h3>Message Analyzed</h3>
                <div className="message-box">{result.message}</div>
              </div>

              {/* Final Verdict */}
              <div className="final-verdict" style={{ borderColor: getRiskColor(result.riskLevel) }}>
                <div className="verdict-header">
                  <span className="verdict-icon">{getRiskIcon(result.riskLevel)}</span>
                  <h2>Final Risk Level: {result.riskLevel}</h2>
                </div>
                <div className="verdict-score">
                  <div className="score-label">Consensus Score</div>
                  <div className="score-value" style={{ color: getRiskColor(result.riskLevel) }}>
                    {(result.details.risk_score * 100).toFixed(1)}%
                  </div>
                </div>
                {result.isCrisis && (
                  <div className="crisis-badge">⚠️ CRISIS PROTOCOL ACTIVATED</div>
                )}
              </div>

              {/* Component Scores */}
              <div className="component-scores">
                <h3>Component Analysis</h3>
                
                {/* Regex Strategy */}
                <div className="component-card regex">
                  <div className="component-header">
                    <span className="component-icon">🔍</span>
                    <h4>Regex Detection</h4>
                    <span className="component-weight">Weight: 40%</span>
                  </div>
                  <div className="component-score">
                    <div className="score-bar">
                      <div 
                        className="score-fill regex-fill" 
                        style={{ width: `${result.details.regex_score * 100}%` }}
                      ></div>
                    </div>
                    <span className="score-text">{(result.details.regex_score * 100).toFixed(1)}%</span>
                  </div>
                  {result.details.matched_patterns && result.details.matched_patterns.length > 0 && (
                    <div className="matched-patterns">
                      <strong>Matched Patterns:</strong>
                      <div className="pattern-tags">
                        {result.details.matched_patterns.map((pattern, idx) => (
                          <span key={idx} className="pattern-tag">{pattern}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Semantic Strategy */}
                <div className="component-card semantic">
                  <div className="component-header">
                    <span className="component-icon">🧬</span>
                    <h4>Semantic Analysis</h4>
                    <span className="component-weight">Weight: 20%</span>
                  </div>
                  <div className="component-score">
                    <div className="score-bar">
                      <div 
                        className="score-fill semantic-fill" 
                        style={{ width: `${result.details.semantic_score * 100}%` }}
                      ></div>
                    </div>
                    <span className="score-text">{(result.details.semantic_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="component-description">
                    Embedding similarity to known crisis patterns
                  </div>
                </div>

                {/* Mistral Strategy */}
                <div className="component-card mistral">
                  <div className="component-header">
                    <span className="component-icon">🤖</span>
                    <h4>Mistral Reasoner</h4>
                    <span className="component-weight">Weight: 30%</span>
                  </div>
                  <div className="component-score">
                    <div className="score-bar">
                      <div 
                        className="score-fill mistral-fill" 
                        style={{ width: `${(result.details.mistral_score || 0) * 100}%` }}
                      ></div>
                    </div>
                    <span className="score-text">
                      {result.details.mistral_score 
                        ? `${(result.details.mistral_score * 100).toFixed(1)}%`
                        : 'Timeout'}
                    </span>
                  </div>
                  {result.details.timeout_occurred && (
                    <div className="timeout-notice">
                      ⏱️ Analysis timed out (graceful degradation)
                    </div>
                  )}
                </div>
              </div>

              {/* Reasoning Trace */}
              <div className="reasoning-section">
                <h3>Reasoning Trace</h3>
                <div className="reasoning-box">
                  {result.details.reasoning}
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="metrics-section">
                <h3>Performance Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span className="metric-label">Total Latency</span>
                    <span className="metric-value">{result.details.latency_ms}ms</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Conversation ID</span>
                    <span className="metric-value">#{result.conversationId}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Timestamp</span>
                    <span className="metric-value">{new Date(result.details.created_at).toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>

              {/* System Response */}
              <div className="response-section">
                <h3>System Response to Student</h3>
                <div className="response-box">
                  {result.response}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConsensusDemo;
