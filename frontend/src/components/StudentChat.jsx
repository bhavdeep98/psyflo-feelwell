import { useState, useRef, useEffect } from 'react'
import './StudentChat.css'

const API_URL = 'http://localhost:8000'

function StudentChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm here to listen. How are you feeling today?",
      timestamp: new Date()
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId] = useState(() => `session_${Date.now()}`)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (e) => {
    e.preventDefault()
    
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    
    // Add user message
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    }])

    setLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage
        })
      })

      if (!response.ok) {
        throw new Error('Failed to send message')
      }

      const data = await response.json()

      // Add assistant response
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        risk_level: data.risk_level,
        is_crisis: data.is_crisis
      }])

    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "I'm having trouble connecting right now. Please try again or reach out to a counselor directly.",
        timestamp: new Date(),
        error: true
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="student-chat">
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div 
              key={idx} 
              className={`message ${msg.role} ${msg.is_crisis ? 'crisis' : ''} ${msg.error ? 'error' : ''}`}
            >
              <div className="message-content">
                {msg.content}
              </div>
              <div className="message-time">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
            autoFocus
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>

      <div className="help-sidebar">
        <h3>Need Help Now?</h3>
        <div className="crisis-resources">
          <div className="resource">
            <span className="icon">📞</span>
            <div>
              <strong>Call 988</strong>
              <p>Suicide & Crisis Lifeline</p>
            </div>
          </div>
          <div className="resource">
            <span className="icon">💬</span>
            <div>
              <strong>Text HOME to 741741</strong>
              <p>Crisis Text Line</p>
            </div>
          </div>
          <div className="resource">
            <span className="icon">🌐</span>
            <div>
              <strong>Online Chat</strong>
              <p>suicidepreventionlifeline.org</p>
            </div>
          </div>
        </div>
        
        <div className="privacy-note">
          <p><strong>Your Privacy</strong></p>
          <p>Your conversations are private. If we detect you might be in crisis, we'll notify your school counselor so they can help.</p>
        </div>
      </div>
    </div>
  )
}

export default StudentChat
