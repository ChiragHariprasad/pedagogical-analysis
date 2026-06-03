import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

const DIMENSIONS = ['effectiveness', 'engagement', 'clarity', 'relevance'];
const DIM_LABELS = { effectiveness: '📊 Effect.', engagement: '🎯 Engage.', clarity: '💡 Clarity', relevance: '🔗 Relev.' };

function Confetti() {
  const colors = ['#6366f1', '#a855f7', '#ec4899', '#10b981', '#f59e0b', '#3b82f6'];
  const pieces = Array.from({ length: 60 }, (_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    color: colors[Math.floor(Math.random() * colors.length)],
    delay: `${Math.random() * 2}s`,
    duration: `${2 + Math.random() * 2}s`,
    size: `${6 + Math.random() * 8}px`,
    rotation: `${Math.random() * 360}deg`,
  }));

  return (
    <div className="confetti-container" id="confetti-container">
      {pieces.map((p) => (
        <div
          key={p.id}
          className="confetti-piece"
          style={{
            left: p.left,
            width: p.size,
            height: p.size,
            backgroundColor: p.color,
            borderRadius: Math.random() > 0.5 ? '50%' : '2px',
            animationDelay: p.delay,
            animationDuration: p.duration,
            transform: `rotate(${p.rotation})`,
          }}
        />
      ))}
    </div>
  );
}

export default function SubmitPage({ pedagogies, ratings, onSubmit, authToken }) {
  const navigate = useNavigate();
  const [submitState, setSubmitState] = useState('idle'); // idle | loading | success | error | already_submitted
  const [errorMsg, setErrorMsg] = useState('');

  const completedCount = pedagogies.filter((p) => {
    const r = ratings[p.id];
    if (!r) return false;
    return (r.effectiveness || r.engagement || r.clarity || r.relevance) && (r.feedback || '').length >= 10;
  }).length;

  const canSubmit = completedCount >= 3;

  const getRatingClass = (val) => {
    if (!val || val === 0) return 'rating-none';
    if (val >= 4) return 'rating-high';
    if (val === 3) return 'rating-mid';
    return 'rating-low';
  };

  const handleSubmit = async () => {
    setSubmitState('loading');
    setErrorMsg('');

    try {
      const submissions = [];

      for (const pedagogy of pedagogies) {
        const r = ratings[pedagogy.id];
        if (!r) continue;

        const hasRating = r.effectiveness || r.engagement || r.clarity || r.relevance;
        const hasFeedback = (r.feedback || '').length >= 10;
        if (!hasRating && !hasFeedback) continue;

        submissions.push({
          pedagogy_id: pedagogy.id,
          pedagogy_name: pedagogy.name,
          effectiveness: r.effectiveness || 0,
          engagement: r.engagement || 0,
          clarity: r.clarity || 0,
          relevance: r.relevance || 0,
          feedback: r.feedback || '',
        });
      }

      const headers = { 'Content-Type': 'application/json' };
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const response = await fetch(`${API_BASE}/api/survey/submit`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ responses: submissions }),
      });

      if (response.status === 403) {
        setSubmitState('already_submitted');
        return;
      }

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      setSubmitState('success');
      if (onSubmit) onSubmit();
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong');
      setSubmitState('error');
    }
  };

  if (submitState === 'loading') {
    return (
      <div className="submit-page" id="submit-page">
        <div className="glass-card-elevated">
          <div className="submit-loading" id="submit-loading">
            <div className="submit-loading-spinner" />
            <p className="submit-loading-text">Processing your feedback through the ABSA engine...</p>
          </div>
        </div>
      </div>
    );
  }

  if (submitState === 'success') {
    return (
      <div className="submit-page" id="submit-page">
        <Confetti />
        <div className="glass-card-elevated">
          <div className="submit-success" id="submit-success">
            <div className="submit-success-icon">✓</div>
            <h3>Questionnaire Submitted Successfully!</h3>
            <p>Thank you for your valuable feedback. Your responses will help improve teaching methodologies through our ABSA analysis engine.</p>
            <button
              className="btn-primary"
              id="submit-go-home-btn"
              onClick={() => navigate('/')}
            >
              <span>Back to Home</span>
              <span>→</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (submitState === 'already_submitted') {
    return (
      <div className="submit-page" id="submit-page">
        <div className="glass-card-elevated">
          <div className="submit-error" id="submit-already-submitted">
            <div className="submit-error-icon">🔒</div>
            <h3 style={{ color: 'var(--warning)' }}>Already Submitted</h3>
            <p>You have already submitted this survey. Each student is allowed only one submission to ensure fairness.</p>
            <button
              className="btn-primary"
              id="submit-home-btn"
              onClick={() => navigate('/')}
            >
              <span>Back to Home</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (submitState === 'error') {
    return (
      <div className="submit-page" id="submit-page">
        <div className="glass-card-elevated">
          <div className="submit-error" id="submit-error">
            <div className="submit-error-icon">⚠️</div>
            <h3>Submission Failed</h3>
            <p>{errorMsg || 'Could not connect to the server. Please ensure the backend is running.'}</p>
            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button className="btn-primary" id="submit-retry-btn" onClick={handleSubmit}>
                <span>Retry</span>
              </button>
              <button className="btn-secondary" id="submit-back-btn" onClick={() => setSubmitState('idle')}>
                <span>Go Back</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="submit-page" id="submit-page">
      <h2 className="gradient-text" id="submit-title">Review & Submit</h2>
      <p className="submit-subtitle" id="submit-subtitle">
        Review your ratings below before submitting. You rated {completedCount} out of {pedagogies.length} pedagogies.
      </p>

      <div className="glass-card-elevated" style={{ padding: '0', overflow: 'hidden' }}>
        <div className="ratings-summary-grid" id="ratings-summary-grid">
          {/* Header row */}
          <div className="ratings-summary-header">Pedagogy</div>
          {DIMENSIONS.map((d) => (
            <div key={d} className="ratings-summary-header">
              {DIM_LABELS[d]}
            </div>
          ))}

          {/* Data rows */}
          {pedagogies.map((p) => {
            const r = ratings[p.id] || {};
            return [
              <div key={`${p.id}-name`} className="ratings-summary-pedagogy">
                <span>{p.emoji}</span>
                <span>{p.name.length > 20 ? p.name.substring(0, 18) + '…' : p.name}</span>
              </div>,
              ...DIMENSIONS.map((d) => (
                <div
                  key={`${p.id}-${d}`}
                  className={`ratings-summary-cell ${getRatingClass(r[d])}`}
                  id={`summary-cell-${p.id}-${d}`}
                >
                  {r[d] ? `${r[d]} ★` : '—'}
                </div>
              )),
            ];
          })}
        </div>
      </div>

      <div className="submit-actions" id="submit-actions">
        {!canSubmit && (
          <div className="submit-warning" id="submit-warning">
            <span>⚠️</span>
            <span>Please complete at least 3 pedagogies (with ratings + 10 chars feedback) to submit.</span>
          </div>
        )}

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button
            className="btn-secondary"
            id="submit-back-to-survey-btn"
            onClick={() => navigate('/survey')}
          >
            ← Back to Questionnaire
          </button>
          <button
            className="btn-primary btn-large"
            id="submit-survey-btn"
            disabled={!canSubmit}
            onClick={handleSubmit}
          >
            <span>Submit Questionnaire</span>
            <span>🚀</span>
          </button>
        </div>
      </div>
    </div>
  );
}
