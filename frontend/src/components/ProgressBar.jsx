const STEP_LABELS = [
  'Welcome',
  'Lecture',
  'PBL',
  'Flipped',
  'Collab',
  'Inquiry',
  'Labs',
  'Submit',
];

export default function ProgressBar({ currentStep = 0, totalSteps = 8 }) {
  const fillPercent =
    totalSteps > 1
      ? (currentStep / (totalSteps - 1)) * 100
      : 0;

  /* Width of the line area = full track minus the two end circles' insets */
  const lineWidth = `calc(100% - 48px)`;

  return (
    <div className="progress-bar" id="progress-bar">
      <div className="progress-bar-track">
        {/* Background line */}
        <div className="progress-bar-line" />

        {/* Filled line */}
        <div
          className="progress-bar-fill"
          style={{
            width: `calc(${fillPercent}% * (1 - 48px / 100%))`,
            maxWidth: lineWidth,
          }}
          id="progress-bar-fill"
        />

        {/* Steps */}
        <div className="progress-steps">
          {STEP_LABELS.map((label, i) => {
            let stepClass = 'progress-step';
            if (i < currentStep) stepClass += ' completed';
            else if (i === currentStep) stepClass += ' active';

            return (
              <div key={i} className={stepClass} id={`progress-step-${i}`}>
                <div className="progress-step-circle">
                  {i < currentStep ? '✓' : i + 1}
                </div>
                <span className="progress-step-label">{label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
