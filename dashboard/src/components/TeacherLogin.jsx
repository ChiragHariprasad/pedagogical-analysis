import { useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '');

export default function TeacherLogin({ onLoginSuccess }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim()) { setError('Please enter the teacher password.'); return; }
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE}/api/teacher/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        const data = await res.json();
        sessionStorage.setItem('teacher_token', data.token);
        onLoginSuccess(data.token);
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.detail || 'Invalid password. Please try again.');
      }
    } catch {
      setError('Cannot connect to server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="teacher-login-container" id="teacher-login-page">
      <div className="teacher-login-card">
        <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>🔐</div>
        <h2>Pedagogical Intelligence Dashboard</h2>
        <p className="teacher-login-subtitle">Teacher Access</p>
        <p className="teacher-login-desc">
          Enter the teacher password to access the analytics dashboard with AI-powered insights.
        </p>

        <form onSubmit={handleSubmit}>
          <input
            type="password"
            className="teacher-login-input"
            id="teacher-password-input"
            placeholder="Enter teacher password..."
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
          />
          {error && <div className="teacher-login-error" id="teacher-login-error">⚠️ {error}</div>}
          <button type="submit" className="teacher-login-submit" id="teacher-login-submit" disabled={loading}>
            {loading ? 'Authenticating...' : '🔓 Access Dashboard'}
          </button>
        </form>

        <p className="teacher-login-footer">
          RV College of Engineering, Bengaluru · Department of AI & ML
        </p>
      </div>
    </div>
  );
}
