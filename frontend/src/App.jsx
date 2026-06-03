import { useState, useCallback } from 'react';
import { Routes, Route, NavLink, useNavigate, useLocation, Navigate } from 'react-router-dom';
import './index.css';
import LoginPage from './components/LoginPage';
import WelcomePage from './components/WelcomePage';
import SurveyForm from './components/SurveyForm';
import ProgressBar from './components/ProgressBar';
import SubmitPage from './components/SubmitPage';

const PEDAGOGIES = [
  { id: 'traditional_lecture', name: 'Traditional Lecture', description: 'Instructor-led direct instruction with PowerPoint/whiteboard presentations', emoji: '🎓', color: 'hsl(220, 70%, 60%)' },
  { id: 'project_based', name: 'Project-Based Learning (PBL)', description: 'Deep, hands-on exploration of real-world problems over extended periods', emoji: '🛠️', color: 'hsl(160, 70%, 50%)' },
  { id: 'flipped_classroom', name: 'Flipped Classroom', description: 'Pre-class materials (videos/readings), class time for active application & discussion', emoji: '🔄', color: 'hsl(280, 70%, 60%)' },
  { id: 'collaborative', name: 'Collaborative / Peer Learning', description: 'Group work, pair programming, peer code reviews, shared responsibility', emoji: '👥', color: 'hsl(30, 80%, 55%)' },
  { id: 'inquiry_based', name: 'Inquiry / Problem-Based Learning', description: 'Learning driven by open-ended questions and problems requiring investigation', emoji: '🔍', color: 'hsl(340, 70%, 60%)' },
  { id: 'experiential_labs', name: 'Experiential / Hands-On Labs', description: 'Direct experience through labs, practical coding exercises, and demos', emoji: '🧪', color: 'hsl(50, 80%, 50%)' },
];

function buildInitialRatings() {
  const initial = {};
  PEDAGOGIES.forEach((p) => {
    initial[p.id] = { effectiveness: 0, engagement: 0, clarity: 0, relevance: 0, feedback: '' };
  });
  return initial;
}

/* ── Multi-Step Survey Page ── */
function SurveyPage({ ratings, onRatingChange }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [animDirection, setAnimDirection] = useState('forward');
  const navigate = useNavigate();
  const pedagogy = PEDAGOGIES[currentStep];

  const handleNext = () => {
    if (currentStep < PEDAGOGIES.length - 1) {
      setAnimDirection('forward');
      setCurrentStep((s) => s + 1);
    } else {
      navigate('/submit');
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setAnimDirection('back');
      setCurrentStep((s) => s - 1);
    } else {
      navigate('/');
    }
  };

  const handleSkip = () => {
    if (currentStep < PEDAGOGIES.length - 1) {
      setAnimDirection('forward');
      setCurrentStep((s) => s + 1);
    } else {
      navigate('/submit');
    }
  };

  /* Progress bar shows: Welcome(0) + 6 pedagogies(1-6) + Submit(7) = 8 steps total.
     currentStep here is 0-5 (pedagogy index), so progressStep = currentStep + 1 */
  const progressStep = currentStep + 1;

  return (
    <div className="survey-step-container" id="survey-step-container">
      <ProgressBar currentStep={progressStep} totalSteps={8} />

      <div key={pedagogy.id}>
        <SurveyForm
          pedagogy={pedagogy}
          ratings={ratings[pedagogy.id]}
          onChange={onRatingChange}
          animDirection={animDirection}
        />
      </div>

      <div className="survey-step-nav" id="survey-step-nav">
        <button className="btn-secondary" id="survey-back-btn" onClick={handleBack}>
          ← {currentStep === 0 ? 'Home' : 'Previous'}
        </button>

        <span className="step-counter" id="survey-step-counter">
          {currentStep + 1} / {PEDAGOGIES.length}
        </span>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn-secondary"
            id="survey-skip-btn"
            onClick={handleSkip}
            title="Skip this pedagogy"
          >
            Skip
          </button>
          <button className="btn-primary" id="survey-next-btn" onClick={handleNext}>
            <span>{currentStep === PEDAGOGIES.length - 1 ? 'Review & Submit' : 'Next'}</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Navbar ── */
function Navbar({ user, onLogout }) {
  const location = useLocation();

  return (
    <nav className="navbar" id="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand" id="navbar-brand">
          <div className="navbar-brand-icon">🧠</div>
          <span>PedagogiQ</span>
        </NavLink>

        <div className="navbar-links" id="navbar-links">
          <NavLink
            to="/survey"
            className={({ isActive }) => `navbar-link ${isActive || location.pathname === '/submit' ? 'active' : ''}`}
            id="navbar-link-survey"
          >
            📝 Questionnaire
          </NavLink>

          {user && (
            <div className="navbar-user-section" id="navbar-user-section">
              <span className="navbar-user-name" id="navbar-user-name">
                👤 {user.name || user.email}
              </span>
              <button
                className="btn-icon"
                id="navbar-logout-btn"
                onClick={onLogout}
                title="Sign out"
              >
                🚪
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

/* ── Protected Route Wrapper ── */
function RequireAuth({ isLoggedIn, children }) {
  if (!isLoggedIn) {
    return <Navigate to="/" replace />;
  }
  return children;
}

/* ── App Root ── */
export default function App() {
  const [ratings, setRatings] = useState(buildInitialRatings);
  const [user, setUser] = useState(() => {
    // Restore session on refresh
    const token = sessionStorage.getItem('auth_token');
    const name = sessionStorage.getItem('auth_name');
    const email = sessionStorage.getItem('auth_email');
    if (token) return { token, name: name || '', email: email || '' };
    return null;
  });

  const isLoggedIn = !!user;

  const handleLoginSuccess = useCallback((userData) => {
    setUser(userData);
    sessionStorage.setItem('auth_token', userData.token);
    sessionStorage.setItem('auth_name', userData.name);
    sessionStorage.setItem('auth_email', userData.email);
  }, []);

  const handleLogout = useCallback(() => {
    setUser(null);
    sessionStorage.removeItem('auth_token');
    sessionStorage.removeItem('auth_name');
    sessionStorage.removeItem('auth_email');
  }, []);

  const handleRatingChange = useCallback((pedagogyId, field, value) => {
    setRatings((prev) => ({
      ...prev,
      [pedagogyId]: {
        ...prev[pedagogyId],
        [field]: value,
      },
    }));
  }, []);

  const handleSubmitComplete = useCallback(() => {
    // Optionally reset after submission
  }, []);

  return (
    <>
      <Navbar user={user} onLogout={handleLogout} />
      <Routes>
        <Route
          path="/"
          element={
            isLoggedIn
              ? <WelcomePage userName={user.name} />
              : <LoginPage onLoginSuccess={handleLoginSuccess} />
          }
        />
        <Route
          path="/survey"
          element={
            <RequireAuth isLoggedIn={isLoggedIn}>
              <SurveyPage ratings={ratings} onRatingChange={handleRatingChange} />
            </RequireAuth>
          }
        />
        <Route
          path="/submit"
          element={
            <RequireAuth isLoggedIn={isLoggedIn}>
              <SubmitPage
                pedagogies={PEDAGOGIES}
                ratings={ratings}
                onSubmit={handleSubmitComplete}
                authToken={user?.token}
              />
            </RequireAuth>
          }
        />
      </Routes>
    </>
  );
}
