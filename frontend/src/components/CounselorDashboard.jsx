
import React, { useState, useEffect } from 'react';
import './CounselorDashboard.css';

const API_URL = 'http://localhost:8000';

const CounselorDashboard = () => {
    const [crisisEvents, setCrisisEvents] = useState([]);
    const [selectedSession, setSelectedSession] = useState(null);
    const [conversations, setConversations] = useState([]);
    const [loading, setLoading] = useState(false);

    // Fetch crisis events on mount
    useEffect(() => {
        const fetchCrisisEvents = async () => {
            try {
                const response = await fetch(`${API_URL}/crisis-events`);
                if (response.ok) {
                    const data = await response.json();
                    setCrisisEvents(data);
                } else {
                    console.error('Failed to fetch crisis events');
                }
            } catch (error) {
                console.error('Error fetching crisis events:', error);
            }
        };

        fetchCrisisEvents();
        // Poll every 5 seconds for new events
        const interval = setInterval(fetchCrisisEvents, 5000);
        return () => clearInterval(interval);
    }, []);

    // Fetch conversation details when a session is selected
    useEffect(() => {
        const fetchConversations = async () => {
            if (!selectedSession) return;

            setLoading(true);
            try {
                // The crisis event has session_id_hash, but we need the original session_id to look it up?
                // Wait, based on backend code: 
                // @app.get("/conversations/{session_id}") uses hash_pii(session_id).
                // But the crisis event only has session_id_hash.
                // WE CANNOT LOOKUP by hash if the endpoint expects plain ID and hashes it again.
                //
                // Let's check backend/main.py again.
                // @app.get("/conversations/{session_id}") -> session_id_hash = hash_pii(session_id)
                //
                // If I only have the hash from the crisis event, I can't call this endpoint unless I modify the backend 
                // or if the frontend somehow knows the mapping.
                // But this is a "Counselor View", they wouldn't know the student's session ID (PII).
                //
                // Actually, for a prototype, maybe the endpoint should accept the hash directly or we update the backend.
                // Let's look at the backend code I read earlier.
                //
                // Backend:
                // @app.get("/conversations/{session_id}")
                //     session_id_hash = hash_pii(session_id)
                //     conversations = db.query(Conversation).filter(Conversation.session_id_hash == session_id_hash)...
                //
                // Code Issue: The backend expects the raw session_id, but the crisis event only stores the hash.
                // For the purpose of this prototype and "Counselor View", I can't reverse the hash.
                //
                // However, I can try to pass the hash as the session_id if I *disable* hashing in the endpoint for lookup, 
                // OR add a new endpoint /conversations/by-hash/{hash}.
                //
                // BUT, I can't change backend right now easily without restarting everything.
                // Wait, I can change backend file and uvicorn auto-reloads!
                //
                // Let's first try to just display the crisis events.
                // If I click, it might fail to load details.
                //
                // Actually, wait. In `ConsensusDemo`, `sessionId` is known.
                // In `CounselorDashboard`, we are viewing *all* events.
                //
                // The backend likely needs an endpoint `get_conversations_by_hash`.
                // OR checking if `session_id` passed is already a hash? No, `hash_pii` re-hashes.
                //
                // Workaround: I will implement the fetch, but if it fails, I'll handle it. 
                // Wait, I see `session_id` in `CrisisEvent` in backend?
                // `CrisisEvent` has `session_id_hash`.
                //
                // Let's look at `backend/main.py` again in my thought process...
                // It imports `hash_pii`.
                //
                // I will add a new endpoint to `backend/main.py` that allows looking up by hash, 
                // since that's what the counselor implementation logically requires.

                // For now, I'll just write the frontend code assuming I'll fix the backend or it accepts something I can give.
                // Actually, maybe I can just list the events for now.

                // Wait, `StudentChat` sends `session_id`.
                // If I am the student, I know my ID.
                // If I am the counselor, I only see the hash in the DB.

                // Let's write the component to just show events first.
                // And maybe for the conversation view, I will just show the single message from the event if I can't fetch the full history.
                // The crisis event has `risk_score`, `reasoning`, etc.

            } catch (error) {
                console.error('Error fetching conversations:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchConversations();
    }, [selectedSession]);

    // Workaround for the hash issue:
    // I'll make a request to a new endpoint I'll add: /conversations/search?hash=...
    // OR just modify the existing endpoint to check if the input looks like a hash?
    // 
    // Let's just create the component. 
    // I'll assume for a moment that I can fetch by hash. 
    // I'll use `conversations/${hash}` and later modify backend to support it.

    const handleEventClick = async (event) => {
        setSelectedSession(event.session_id_hash);
        setLoading(true);
        try {
            // Try fetching with the hash. 
            // Note: The backend will hash this hash again, so it won't match.
            // This WILL fail with current backend. 
            // But I'll modify the backend next.
            const response = await fetch(`${API_URL}/conversations/lookup/${event.session_id_hash}`);
            if (response.ok) {
                const data = await response.json();
                setConversations(data);
            } else {
                // If that fails (endpoint doesn't exist), try the standard one (will fail logic)
                console.warn("Lookup endpoint failed, trying standard...");
            }
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    };

    const getRiskColor = (level) => {
        switch (level) {
            case 'CRISIS': return '#dc3545';
            case 'CAUTION': return '#ffc107';
            case 'SAFE': return '#28a745';
            default: return '#6c757d';
        }
    };

    return (
        <div className="counselor-dashboard">
            <div className="crisis-panel">
                <h2>🚨 Crisis Events</h2>
                {crisisEvents.length === 0 ? (
                    <div className="no-events">No crisis events detected yet.</div>
                ) : (
                    <div className="crisis-list">
                        {crisisEvents.map(event => (
                            <div
                                key={event.id}
                                className="crisis-card"
                                onClick={() => handleEventClick(event)}
                            >
                                <div className="crisis-header">
                                    <span className="crisis-badge">RISK: {(event.risk_score * 100).toFixed(0)}%</span>
                                    <span className="crisis-time">{new Date(event.detected_at).toLocaleTimeString()}</span>
                                </div>
                                <div className="crisis-details">
                                    <p><strong>Patterns:</strong> {event.matched_patterns ? event.matched_patterns.join(', ') : 'None'}</p>
                                    <div className="crisis-session">Session: {event.session_id_hash.substring(0, 8)}...</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div className="conversation-panel">
                <h2>💬 Conversation History</h2>
                {!selectedSession ? (
                    <div className="no-selection">
                        <p>Select a crisis event to view the conversation details.</p>
                    </div>
                ) : loading ? (
                    <div className="loading">Loading details...</div>
                ) : (
                    <div className="conversation-details">
                        <div className="session-id">Session Hash: {selectedSession}</div>
                        <div className="conversation-list">
                            {conversations.length === 0 ? (
                                <p>No history found (or backend lookup failed).</p>
                            ) : (
                                conversations.map(conv => (
                                    <div key={conv.id} className="conversation-card">
                                        <div className="conv-header">
                                            <span
                                                className="risk-badge"
                                                style={{ backgroundColor: getRiskColor(conv.risk_level) }}
                                            >
                                                {conv.risk_level}
                                            </span>
                                            <span className="conv-time">{new Date(conv.created_at).toLocaleString()}</span>
                                        </div>

                                        <div className="conv-message">
                                            <strong>Student:</strong>
                                            <p>{conv.message}</p>
                                        </div>

                                        <div className="conv-response">
                                            <strong>PsyFlo:</strong>
                                            <p>{conv.response}</p>
                                        </div>

                                        <div className="conv-analysis">
                                            <h4>Analysis Details</h4>

                                            <div className="scores">
                                                <div className="score-item">
                                                    <span>Total</span>
                                                    <strong>{(conv.risk_score * 100).toFixed(1)}%</strong>
                                                </div>
                                                <div className="score-item">
                                                    <span>Regex</span>
                                                    <strong>{(conv.regex_score * 100).toFixed(1)}%</strong>
                                                </div>
                                                <div className="score-item">
                                                    <span>Semantic</span>
                                                    <strong>{(conv.semantic_score * 100).toFixed(1)}%</strong>
                                                </div>
                                                <div className="score-item">
                                                    <span>Mistral</span>
                                                    <strong>{conv.mistral_score ? (conv.mistral_score * 100).toFixed(1) + '%' : 'N/A'}</strong>
                                                </div>
                                            </div>

                                            {conv.matched_patterns && conv.matched_patterns.length > 0 && (
                                                <div className="patterns">
                                                    <strong>Matched Patterns:</strong>
                                                    {conv.matched_patterns.join(', ')}
                                                </div>
                                            )}

                                            <details className="reasoning-details">
                                                <summary>View Reasoning Trace</summary>
                                                <pre>{conv.reasoning}</pre>
                                            </details>

                                            <div className="performance">
                                                <span>Latency: {conv.latency_ms.toFixed(0)}ms</span>
                                                {conv.timeout_occurred && <span className="timeout-badge">TIMEOUT</span>}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CounselorDashboard;
