import { useNavigate } from 'react-router-dom';

const PEDAGOGIES = [
  { emoji: '🎓', name: 'Traditional Lecture', color: 'hsl(220, 70%, 60%)' },
  { emoji: '🛠️', name: 'Project-Based Learning', color: 'hsl(160, 70%, 50%)' },
  { emoji: '🔄', name: 'Flipped Classroom', color: 'hsl(280, 70%, 60%)' },
  { emoji: '👥', name: 'Collaborative Learning', color: 'hsl(30, 80%, 55%)' },
  { emoji: '🔍', name: 'Inquiry-Based Learning', color: 'hsl(340, 70%, 60%)' },
  { emoji: '🧪', name: 'Experiential Labs', color: 'hsl(50, 80%, 50%)' },
];

export default function WelcomePage({ userName }) {
  const navigate = useNavigate();

  return (
    <div className="welcome-page">
      {/* Personalized Greeting */}
      {userName && (
        <div className="welcome-greeting" id="welcome-greeting">
          <span className="welcome-greeting-wave">👋</span>
          <span>Welcome, <strong>{userName}</strong></span>
        </div>
      )}

      <h1 className="welcome-title gradient-text" id="welcome-title">
        NLP Course Feedback Questionnaire
      </h1>
      <p className="welcome-subtitle" id="welcome-subtitle">
        Pedagogical Intelligence System — VI Semester
      </p>

      <p className="welcome-description" id="welcome-description">
        This structured questionnaire evaluates <strong>6 teaching methodologies</strong> used
        in your NLP course. For each pedagogy, you will answer 5 carefully designed questions —
        4 Likert-scale ratings and 1 open-ended response. Your honest feedback powers our
        Aspect-Based Sentiment Analysis engine to generate actionable insights for improving
        course delivery. It takes about <strong>5–10 minutes</strong>.
      </p>

      <div className="welcome-pedagogies" id="welcome-pedagogies">
        {PEDAGOGIES.map((p, i) => (
          <div
            key={i}
            className="welcome-pedagogy-chip"
            id={`welcome-chip-${i}`}
            style={{ borderColor: `${p.color}30` }}
          >
            <span className="chip-emoji">{p.emoji}</span>
            {p.name}
          </div>
        ))}
      </div>

      <button
        className="btn-primary btn-large"
        id="welcome-begin-btn"
        onClick={() => navigate('/survey')}
      >
        <span>Begin Questionnaire</span>
        <span>→</span>
      </button>

      <p className="welcome-footer" id="welcome-footer">
        RV College of Engineering • Department of AIML
      </p>
    </div>
  );
}
