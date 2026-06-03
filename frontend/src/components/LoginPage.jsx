import { useState } from 'react';
import { GoogleLogin } from '@react-oauth/google';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export default function LoginPage({ onLoginSuccess }) {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_BASE}/api/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential: credentialResponse.credential }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || 'Authentication failed. Please try again.');
        setLoading(false);
        return;
      }

      if (data.already_submitted) {
        setError('You have already submitted this survey. Each student can only submit once.');
        setLoading(false);
        return;
      }

      sessionStorage.setItem('auth_token', data.token);
      onLoginSuccess({
        token: data.token,
        name: data.name || '',
        email: data.email || '',
      });
    } catch (err) {
      setError('Network error. Please ensure the backend server is running and try again.');
      setLoading(false);
    }
  };

  return (
    <div className="login-page" id="login-page">
      <div className="login-card glass-card-elevated" id="login-card">
        {/* College Branding */}
        <div className="login-branding" id="login-branding">
          <div className="login-college-logo" id="login-college-logo">
            <span className="login-logo-icon">🎓</span>
          </div>
          <h2 className="login-college-name" id="login-college-name">
            RV College of Engineering
          </h2>
          <p className="login-department" id="login-department">
            Department of Artificial Intelligence &amp; Machine Learning
          </p>
        </div>

        {/* Survey Title */}
        <div className="login-title-section" id="login-title-section">
          <h1 className="login-title gradient-text" id="login-title">
            NLP Course Feedback Survey
          </h1>
          <p className="login-subtitle" id="login-subtitle">
            VI Semester — Academic Year 2025–26
          </p>
        </div>

        {/* Instructions */}
        <div className="login-instructions" id="login-instructions">
          <div className="login-instruction-item">
            <span className="login-instruction-icon">🔐</span>
            <span>Sign in with your college email (<strong>@rvce.edu.in</strong>) to begin the survey.</span>
          </div>
          <div className="login-instruction-item">
            <span className="login-instruction-icon">📝</span>
            <span>Each student can submit only <strong>once</strong>. Your responses are confidential.</span>
          </div>
          <div className="login-instruction-item">
            <span className="login-instruction-icon">⏱️</span>
            <span>The questionnaire takes approximately <strong>5–10 minutes</strong> to complete.</span>
          </div>
        </div>

        {/* Google Login Area */}
        <div className="login-form" id="login-form" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', marginTop: '1rem' }}>
          {loading ? (
            <div style={{ color: 'var(--text-light)' }}>Authenticating...</div>
          ) : (
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('Google Login Failed')}
              useOneTap
              theme="filled_blue"
              shape="pill"
              text="continue_with"
            />
          )}

          {/* Error Display */}
          {error && (
            <div className="login-error" id="login-error" style={{ width: '100%', marginTop: '1rem' }}>
              <span className="login-error-icon">⚠️</span>
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="login-footer" id="login-footer">
          Pedagogical Intelligence System • Powered by ABSA &amp; Gemini AI
        </p>
      </div>
    </div>
  );
}
