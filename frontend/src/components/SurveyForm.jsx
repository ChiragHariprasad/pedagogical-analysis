import { useState } from 'react';

const QUESTIONS = [
  {
    key: 'effectiveness',
    number: 'Q1',
    label: 'Effectiveness',
    icon: '📊',
    question: 'How effectively did this teaching method help you understand and retain the course concepts?',
  },
  {
    key: 'engagement',
    number: 'Q2',
    label: 'Engagement',
    icon: '🎯',
    question: 'How engaged and motivated did you feel during sessions using this method?',
  },
  {
    key: 'clarity',
    number: 'Q3',
    label: 'Clarity',
    icon: '💡',
    question: 'How clearly was the content presented and explained using this approach?',
  },
  {
    key: 'relevance',
    number: 'Q4',
    label: 'Relevance',
    icon: '🔗',
    question: 'How relevant was this method to your practical learning goals and future career?',
  },
];

const LIKERT_OPTIONS = [
  { value: 1, label: 'Strongly Disagree', short: 'SD' },
  { value: 2, label: 'Disagree', short: 'D' },
  { value: 3, label: 'Neutral', short: 'N' },
  { value: 4, label: 'Agree', short: 'A' },
  { value: 5, label: 'Strongly Agree', short: 'SA' },
];

function LikertScale({ questionKey, rating, onRate, pedagogyId, accentColor }) {
  const [hoveredValue, setHoveredValue] = useState(null);

  return (
    <div className="likert-scale" id={`likert-${pedagogyId}-${questionKey}`}>
      {LIKERT_OPTIONS.map((option) => {
        const isSelected = rating === option.value;
        const isHovered = hoveredValue === option.value;

        return (
          <button
            key={option.value}
            type="button"
            className={`likert-option ${isSelected ? 'selected' : ''} ${isHovered && !isSelected ? 'hovered' : ''}`}
            id={`likert-${pedagogyId}-${questionKey}-${option.value}`}
            style={{
              '--option-accent': accentColor,
              '--option-value': option.value,
            }}
            onClick={() => onRate(option.value)}
            onMouseEnter={() => setHoveredValue(option.value)}
            onMouseLeave={() => setHoveredValue(null)}
            aria-label={`Rate ${questionKey} as ${option.label}`}
            title={option.label}
          >
            <span className="likert-option-value">{option.value}</span>
            <span className="likert-option-label">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export default function SurveyForm({ pedagogy, ratings, onChange, animDirection }) {
  const currentRatings = ratings || {
    effectiveness: 0,
    engagement: 0,
    clarity: 0,
    relevance: 0,
    feedback: '',
  };

  const feedbackLen = (currentRatings.feedback || '').length;
  let charCountClass = 'feedback-char-count';
  if (feedbackLen > 0 && feedbackLen < 10) charCountClass += ' warning';
  else if (feedbackLen >= 10) charCountClass += ' valid';

  const animClass = animDirection === 'back' ? 'step-back' : 'step-enter';

  return (
    <div
      className={`pedagogy-card ${animClass}`}
      style={{ '--card-accent': pedagogy.color }}
      id={`pedagogy-card-${pedagogy.id}`}
      key={pedagogy.id}
    >
      {/* Pedagogy Header */}
      <div className="pedagogy-card-header">
        <div className="pedagogy-card-emoji" id={`pedagogy-emoji-${pedagogy.id}`}>
          {pedagogy.emoji}
        </div>
        <h2 className="pedagogy-card-title" id={`pedagogy-title-${pedagogy.id}`}>
          {pedagogy.name}
        </h2>
      </div>
      <p className="pedagogy-card-description" id={`pedagogy-desc-${pedagogy.id}`}>
        {pedagogy.description}
      </p>

      {/* Likert Questions Q1–Q4 */}
      <div className="survey-questions-container" id={`questions-${pedagogy.id}`}>
        {QUESTIONS.map((q) => (
          <div
            className="survey-question-block"
            key={q.key}
            id={`question-block-${pedagogy.id}-${q.key}`}
          >
            <div className="survey-question-header">
              <span className="survey-question-number" style={{ color: pedagogy.color }}>
                {q.number}
              </span>
              <span className="survey-question-icon">{q.icon}</span>
            </div>
            <p className="survey-question-text" id={`question-text-${pedagogy.id}-${q.key}`}>
              {q.question}
            </p>
            <LikertScale
              questionKey={q.key}
              rating={currentRatings[q.key]}
              pedagogyId={pedagogy.id}
              accentColor={pedagogy.color}
              onRate={(val) => onChange(pedagogy.id, q.key, val)}
            />
          </div>
        ))}
      </div>

      {/* Q5: Qualitative Feedback */}
      <div
        className="survey-question-block"
        id={`question-block-${pedagogy.id}-feedback`}
      >
        <div className="survey-question-header">
          <span className="survey-question-number" style={{ color: pedagogy.color }}>
            Q5
          </span>
          <span className="survey-question-icon">✍️</span>
        </div>
        <p className="survey-question-text" id={`question-text-${pedagogy.id}-feedback`}>
          In your own words, describe your experience with this teaching method. What aspects did you find most helpful? What improvements would you suggest?
          <span className="survey-question-hint"> (You can write in English, Hindi, or Hinglish)</span>
        </p>

        <div className="feedback-textarea-wrapper" id={`feedback-wrapper-${pedagogy.id}`}>
          <textarea
            className="feedback-textarea"
            id={`feedback-${pedagogy.id}`}
            placeholder={`Share your detailed thoughts about ${pedagogy.name}...\n\nFor example: "The ${pedagogy.name.toLowerCase()} sessions were very helpful because..." or mix languages: "Lectures were clear but thoda boring tha, more interactive examples chahiye."`}
            value={currentRatings.feedback || ''}
            onChange={(e) => onChange(pedagogy.id, 'feedback', e.target.value)}
            rows={5}
          />
          <div className={charCountClass} id={`char-count-${pedagogy.id}`}>
            {feedbackLen === 0
              ? 'Minimum 10 characters required'
              : feedbackLen < 10
              ? `${feedbackLen}/10 — need ${10 - feedbackLen} more characters`
              : `${feedbackLen} characters ✓`}
          </div>
        </div>
      </div>
    </div>
  );
}
