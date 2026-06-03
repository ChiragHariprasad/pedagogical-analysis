import { useState } from 'react';

export default function LoginPage({ onLoginSuccess }) {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedEmail = email.trim().toLowerCase();
    const trimmedName = name.trim();

    if (!trimmedEmail) {
      setError('Please enter your college email address.');
      return;
    }
    if (!trimmedEmail.endsWith('@rvce.edu.in')) {
      setError('Only @rvce.edu.in college email addresses are allowed.');
      return;
    }
    if (!trimmedName) {
      setError('Please enter your full name.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const res = await fetch('http://localhost:8000/api/auth/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: trimmedEmail, name: trimmedName }),
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

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="login-form" id="login-form">
          <div className="login-input-group">
            <label htmlFor="login-name-input" className="login-input-label">Full Name</label>
            <input
              type="text"
              className="login-input"
              id="login-name-input"
              placeholder="e.g. Manoj Malipatil"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="login-input-group">
            <label htmlFor="login-email-input" className="login-input-label">College Email</label>
            <input
              type="email"
              className="login-input"
              id="login-email-input"
              placeholder="e.g. yourname.aiml21@rvce.edu.in"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {/* Error Display */}
          {error && (
            <div className="login-error" id="login-error">
              <span className="login-error-icon">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            className="login-submit-btn"
            id="login-submit-btn"
            disabled={loading}
          >
            {loading ? (
              <span>Authenticating...</span>
            ) : (
              <>
                <span>🔓</span>
                <span>Sign In & Start Survey</span>
              </>
            )}
          </button>
        </form>

        {/* Footer */}
        <p className="login-footer" id="login-footer">
          Pedagogical Intelligence System • Powered by ABSA &amp; Gemini AI
        </p>
      </div>
    </div>
  );
}
