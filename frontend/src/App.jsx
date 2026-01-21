import { useState } from 'react'
import StudentChat from './components/StudentChat'
import CounselorDashboard from './components/CounselorDashboard'
import './App.css'

function App() {
  const [view, setView] = useState('student') // 'student' or 'counselor'

  return (
    <div className="app">
      <header className="app-header">
        <h1>PsyFlo</h1>
        <p className="tagline">Mental Health Support</p>
        <div className="view-toggle">
          <button 
            className={view === 'student' ? 'active' : ''}
            onClick={() => setView('student')}
          >
            Student Chat
          </button>
          <button 
            className={view === 'counselor' ? 'active' : ''}
            onClick={() => setView('counselor')}
          >
            Counselor Dashboard
          </button>
        </div>
      </header>

      <main className="app-main">
        {view === 'student' ? <StudentChat /> : <CounselorDashboard />}
      </main>

      <footer className="app-footer">
        <p>🆘 Crisis? Call 988 or text HOME to 741741</p>
      </footer>
    </div>
  )
}

export default App
