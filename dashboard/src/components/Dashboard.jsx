import { useState, useEffect, useCallback } from 'react';
import Plot from './PlotWrapper';

const PEDAGOGY_META = {
  traditional_lecture: { name: 'Traditional Lecture', icon: '🎓' },
  project_based: { name: 'Project-Based Learning', icon: '🛠️' },
  flipped_classroom: { name: 'Flipped Classroom', icon: '🔄' },
  collaborative: { name: 'Collaborative Learning', icon: '👥' },
  inquiry_based: { name: 'Inquiry-Based Learning', icon: '🔍' },
  experiential_labs: { name: 'Experiential Labs', icon: '🧪' },
};

const DIMENSIONS = ['effectiveness', 'engagement', 'clarity', 'relevance'];
const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '');

const NAV_ITEMS = [
  { id: 'overview', icon: '📊', label: 'Dashboard' },
  { id: 'aspects', icon: '📋', label: 'Aspect Analysis' },
  { id: 'sentiment', icon: '📈', label: 'Sentiment Trends' },
  { id: 'recommendations', icon: '💡', label: 'Pedagogical Recommendations' },
  { id: 'feedback', icon: '💬', label: 'Student Feedback' },
];

export default function Dashboard({ onLogout }) {
  const [analytics, setAnalytics] = useState(null);
  const [responses, setResponses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeView, setActiveView] = useState('overview');
  const [geminiSummaries, setGeminiSummaries] = useState({});
  const [geminiLoading, setGeminiLoading] = useState({});
  const [interventionData, setInterventionData] = useState(null);
  const [interventionLoading, setInterventionLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [aRes, rRes] = await Promise.all([
        fetch(`${API_BASE}/api/analytics`),
        fetch(`${API_BASE}/api/responses`),
      ]);
      if (!aRes.ok) throw new Error('Failed to fetch analytics');
      const aData = await aRes.json();
      setAnalytics(aData);
      if (rRes.ok) {
        const rData = await rRes.json();
        setResponses(Array.isArray(rData) ? rData : rData.responses || []);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  /* ── Gemini summary fetch ── */
  const fetchGeminiSummary = useCallback(async (pedagogyId) => {
    if (geminiSummaries[pedagogyId] || geminiLoading[pedagogyId]) return;
    setGeminiLoading(prev => ({ ...prev, [pedagogyId]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/summary/${pedagogyId}`);
      if (res.ok) {
        const data = await res.json();
        setGeminiSummaries(prev => ({ ...prev, [pedagogyId]: data.summary || 'No summary available.' }));
      } else {
        setGeminiSummaries(prev => ({ ...prev, [pedagogyId]: 'Summary not available.' }));
      }
    } catch {
      setGeminiSummaries(prev => ({ ...prev, [pedagogyId]: 'Failed to load summary.' }));
    } finally {
      setGeminiLoading(prev => ({ ...prev, [pedagogyId]: false }));
    }
  }, [geminiSummaries, geminiLoading]);

  /* ── Fetch intervention for lowest-scoring pedagogy ── */
  const fetchIntervention = useCallback(async (pedagogyId) => {
    if (interventionData || interventionLoading) return;
    setInterventionLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/summary/${pedagogyId}`);
      if (res.ok) {
        const data = await res.json();
        setInterventionData({ pedagogyId, name: data.pedagogy_name, summary: data.summary });
      }
    } catch { /* silent */ }
    setInterventionLoading(false);
  }, [interventionData, interventionLoading]);

  /* ── Loading / Error states ── */
  if (loading) {
    return (
      <div className="dashboard-layout">
        <div className="dashboard-main" style={{ marginLeft: 0 }}>
          <div className="loading-container">
            <div className="loading-spinner" />
            <p className="loading-text">Loading analytics data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-layout">
        <div className="dashboard-main" style={{ marginLeft: 0 }}>
          <div className="loading-container">
            <div className="empty-state">
              <div className="empty-state-icon">⚠️</div>
              <h3>Could not load dashboard</h3>
              <p>{error}. Ensure backend is running at {API_BASE}.</p>
              <button className="retry-btn" onClick={fetchData}>Retry</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── Normalize data ── */
  const rawList = analytics?.pedagogy_analytics || [];
  const pedagogyStats = Array.isArray(rawList)
    ? rawList.reduce((acc, item) => { acc[item.pedagogy_id] = item; return acc; }, {})
    : rawList;
  const totalResponses = analytics?.total_responses || 0;

  if (Object.keys(pedagogyStats).length === 0) {
    return (
      <div className="dashboard-layout">
        <div className="dashboard-main" style={{ marginLeft: 0 }}>
          <div className="loading-container">
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <h3>No Data Yet</h3>
              <p>No survey responses have been submitted yet.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── Compute aspect scores & find weakest ── */
  const aspectCards = Object.entries(pedagogyStats).map(([pid, stats]) => {
    const meta = PEDAGOGY_META[pid] || { name: pid, icon: '📋' };
    const avgs = DIMENSIONS.map(d => Number(stats[`avg_${d}`]) || 0);
    const overall = avgs.filter(v => v > 0).length > 0
      ? Math.round((avgs.reduce((a, b) => a + b, 0) / avgs.filter(v => v > 0).length / 5) * 100)
      : 0;

    // Pick a representative quote
    const pedResponses = responses.filter(r => r.pedagogy_id === pid);
    const quote = pedResponses.length > 0 ? pedResponses[0].feedback : '';

    return { pid, name: meta.name, icon: meta.icon, score: overall, quote, stats, count: stats.count || 0 };
  }).sort((a, b) => b.score - a.score);

  const weakest = aspectCards[aspectCards.length - 1];

  // Auto-fetch intervention for weakest pedagogy
  if (weakest && !interventionData && !interventionLoading) {
    fetchIntervention(weakest.pid);
  }

  /* ── Global keyword data from top_aspects ── */
  const keywordRows = [];
  Object.entries(pedagogyStats).forEach(([pid, stats]) => {
    const meta = PEDAGOGY_META[pid] || { name: pid };
    (stats.top_aspects || []).forEach(a => {
      keywordRows.push({
        aspect: meta.name,
        keyword: a.aspect || a.name || 'unknown',
        count: a.count || 0,
        sentiment: (a.sentiment || 'neutral').toLowerCase(),
      });
    });
  });
  keywordRows.sort((a, b) => b.count - a.count);

  /* ── Sentiment bar data for trend chart ── */
  const sentimentPedagogies = Object.keys(pedagogyStats);
  const posData = sentimentPedagogies.map(pid => pedagogyStats[pid]?.sentiment_distribution?.Positive || 0);
  const negData = sentimentPedagogies.map(pid => pedagogyStats[pid]?.sentiment_distribution?.Negative || 0);
  const totalPos = posData.reduce((a, b) => a + b, 0);
  const totalNeg = negData.reduce((a, b) => a + b, 0);
  const totalSent = totalPos + totalNeg || 1;
  const overallPositivePct = Math.round((totalPos / totalSent) * 100);

  /* ── Parse intervention recommendations from summary ── */
  const parseRecommendations = (summary) => {
    if (!summary) return [];
    const lines = summary.split('\n').filter(l => l.trim());
    const recs = [];
    lines.forEach(l => {
      const clean = l.replace(/^\*\*.*?\*\*:?\s*/, '').replace(/^[-•*]\s*/, '').trim();
      if (clean.length > 30 && clean.length < 300 && !clean.startsWith('**')) {
        recs.push(clean);
      }
    });
    return recs.slice(0, 3);
  };

  /* ═══════════════════════ RENDER ═══════════════════════ */
  const renderOverview = () => (
    <>
      {/* Aspect Performance Cards */}
      <div className="aspect-cards-grid">
        {aspectCards.map(card => (
          <div
            key={card.pid}
            className={`aspect-card ${card.score < 60 ? 'warning' : ''}`}
            id={`aspect-card-${card.pid}`}
            onClick={() => { setActiveView('aspects'); fetchGeminiSummary(card.pid); }}
          >
            <div className="aspect-card-header">
              <div className="aspect-card-icon">{card.icon}</div>
              <span className="aspect-card-name">{card.name.length > 16 ? card.name.substring(0, 15) + '…' : card.name}</span>
              <span className={`aspect-card-score ${card.score < 60 ? 'warning' : 'good'}`}>
                {card.score}%
              </span>
            </div>
            {card.quote && (
              <p className="aspect-card-quote">"{card.quote}"</p>
            )}
          </div>
        ))}
      </div>

      {/* Intervention + Sentiment Trend */}
      <div className="split-row">
        {/* Pedagogical Intervention Required */}
        <div className="intervention-card">
          <div className="intervention-header">
            <span className="intervention-icon">⚠️</span>
            <h3 className="intervention-title">Pedagogical Intervention Required</h3>
          </div>
          {weakest && (
            <p className="intervention-description">
              Sentiment analysis indicates lower satisfaction with <strong>'{weakest.name}'</strong> (score: {weakest.score}%).
              Targeted interventions are recommended to improve student outcomes.
            </p>
          )}

          {interventionLoading && (
            <div className="ai-summary-loading">Generating AI recommendations...</div>
          )}

          {interventionData && parseRecommendations(interventionData.summary).map((rec, i) => (
            <div className="intervention-item" key={i}>
              <div className="intervention-item-icon">💡</div>
              <div className="intervention-item-content">
                <p className="intervention-item-desc">{rec}</p>
              </div>
              <button className="intervention-apply-btn">Apply</button>
            </div>
          ))}

          {interventionData && parseRecommendations(interventionData.summary).length === 0 && (
            <div className="intervention-item">
              <div className="intervention-item-icon">📝</div>
              <div className="intervention-item-content">
                <p className="intervention-item-desc">{interventionData.summary}</p>
              </div>
            </div>
          )}
        </div>

        {/* Overall Sentiment Trend */}
        <div className="trend-card">
          <h3 className="trend-card-title">Overall Sentiment</h3>
          <p className="trend-card-subtitle">Across all pedagogies</p>
          <Plot
            data={[{
              x: sentimentPedagogies.map(pid => (PEDAGOGY_META[pid]?.name || pid).split(' ')[0]),
              y: posData,
              type: 'bar',
              marker: { color: posData.map((p, i) => p >= negData[i] ? '#c8d5c2' : '#e8c5c5') },
              hoverinfo: 'x+y',
            }]}
            layout={{
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#555', size: 10, family: 'Inter' },
              xaxis: { tickfont: { size: 9 }, gridcolor: 'transparent' },
              yaxis: { visible: false },
              margin: { t: 10, b: 40, l: 10, r: 10 },
              autosize: true, bargap: 0.3,
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '200px' }}
            useResizeHandler
          />
          <div className="trend-big-number">{overallPositivePct}%</div>
          <p className="trend-big-label">Current Positive Sentiment Average</p>
        </div>
      </div>

      {/* Keyword Sentiment Breakdown */}
      <div className="keyword-table-card">
        <h3 className="keyword-table-title">Keyword Sentiment Breakdown</h3>
        <p className="keyword-table-subtitle">High-frequency terms extracted from qualitative feedback.</p>
        <table className="keyword-table">
          <thead>
            <tr>
              <th>Aspect</th>
              <th>Keyword</th>
              <th>Frequency</th>
              <th>Sentiment Polarity</th>
            </tr>
          </thead>
          <tbody>
            {keywordRows.slice(0, 12).map((row, i) => (
              <tr key={i}>
                <td>{row.aspect}</td>
                <td><span className="keyword-text">{row.keyword}</span></td>
                <td>{row.count}</td>
                <td><span className={`sentiment-tag ${row.sentiment}`}>{row.sentiment}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );

  const renderAspectAnalysis = () => (
    <>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', marginBottom: '1.25rem' }}>
        Per-Pedagogy Detailed Analysis
      </h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.85rem' }}>
        Click any pedagogy to load an AI-generated summary from Gemini.
      </p>
      {aspectCards.map(card => {
        const stats = card.stats;
        return (
          <div key={card.pid} className="ai-summary-card" id={`detail-${card.pid}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.25rem' }}>{card.icon}</span>
                <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem' }}>{card.name}</h3>
                <span className={`sentiment-tag ${card.score >= 70 ? 'positive' : card.score >= 50 ? 'neutral' : 'negative'}`}>
                  {card.score}%
                </span>
              </div>
              {!geminiSummaries[card.pid] && !geminiLoading[card.pid] && (
                <button className="action-btn" onClick={() => fetchGeminiSummary(card.pid)}>
                  ✨ Generate AI Summary
                </button>
              )}
            </div>

            {/* Rating bars */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
              {DIMENSIONS.map(d => {
                const val = Number(stats[`avg_${d}`]) || 0;
                return (
                  <div key={d} style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'capitalize', marginBottom: '0.25rem' }}>{d}</div>
                    <div style={{ background: 'var(--bg-page)', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
                      <div style={{ width: `${(val / 5) * 100}%`, height: '100%', background: val >= 3.5 ? 'var(--color-positive)' : val >= 2.5 ? '#f59e0b' : 'var(--color-negative)', borderRadius: '4px', transition: 'width 0.5s ease' }} />
                    </div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, marginTop: '0.2rem' }}>{val.toFixed(1)}/5</div>
                  </div>
                );
              })}
            </div>

            {/* AI Summary */}
            {geminiLoading[card.pid] && <div className="ai-summary-loading">Generating AI summary...</div>}
            {geminiSummaries[card.pid] && (
              <>
                <div className="ai-summary-header">
                  <span>✨</span>
                  <span className="ai-summary-badge">Gemini AI Summary</span>
                </div>
                <p className="ai-summary-text">{geminiSummaries[card.pid]}</p>
              </>
            )}
          </div>
        );
      })}
    </>
  );

  const renderSentimentTrends = () => (
    <>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', marginBottom: '1.25rem' }}>
        Sentiment Distribution
      </h2>
      <div className="ai-summary-card">
        <Plot
          data={[
            {
              x: sentimentPedagogies.map(pid => (PEDAGOGY_META[pid]?.name || pid)),
              y: posData,
              name: 'Positive', type: 'bar',
              marker: { color: '#2e7d32' },
            },
            {
              x: sentimentPedagogies.map(pid => (PEDAGOGY_META[pid]?.name || pid)),
              y: negData,
              name: 'Negative', type: 'bar',
              marker: { color: '#c62828' },
            },
          ]}
          layout={{
            barmode: 'group',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#333', size: 11, family: 'Inter' },
            xaxis: { tickfont: { size: 9 } },
            yaxis: { title: 'Aspect Count', gridcolor: '#eee' },
            legend: { orientation: 'h', y: 1.1 },
            margin: { t: 30, b: 80, l: 50, r: 20 },
            autosize: true, bargap: 0.3,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '400px' }}
          useResizeHandler
        />
      </div>
    </>
  );

  const renderFeedback = () => (
    <>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', marginBottom: '1.25rem' }}>
        Student Feedback ({responses.length})
      </h2>
      <div className="feedback-section">
        {responses.slice(0, 30).map((resp, i) => {
          const meta = PEDAGOGY_META[resp.pedagogy_id] || { name: resp.pedagogy_id };
          return (
            <div className="feedback-item" key={i}>
              <span className="feedback-pedagogy-badge">{meta.name?.split(' ')[0]}</span>
              <p className="feedback-text">{resp.feedback || 'No written feedback'}</p>
              <div className="feedback-rating-pills">
                <span className="feedback-pill">E:{resp.effectiveness || '—'}</span>
                <span className="feedback-pill">G:{resp.engagement || '—'}</span>
                <span className="feedback-pill">C:{resp.clarity || '—'}</span>
                <span className="feedback-pill">R:{resp.relevance || '—'}</span>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );

  const renderRecommendations = () => (
    <>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.4rem', marginBottom: '0.5rem' }}>
        Pedagogical Recommendations
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
        AI-generated recommendations based on student feedback analysis. Click "Generate" to create a summary for any pedagogy.
      </p>
      {aspectCards.map(card => (
        <div key={card.pid} className="ai-summary-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '1.1rem' }}>{card.icon}</span>
              <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1rem' }}>{card.name}</h3>
              <span className={`sentiment-tag ${card.score >= 70 ? 'positive' : card.score >= 50 ? 'neutral' : 'negative'}`}>
                {card.score}%
              </span>
            </div>
            {!geminiSummaries[card.pid] && !geminiLoading[card.pid] && (
              <button className="action-btn" onClick={() => fetchGeminiSummary(card.pid)}>
                ✨ Generate
              </button>
            )}
          </div>
          {geminiLoading[card.pid] && <div className="ai-summary-loading">Generating...</div>}
          {geminiSummaries[card.pid] && (
            <p className="ai-summary-text">{geminiSummaries[card.pid]}</p>
          )}
        </div>
      ))}
    </>
  );

  const viewRenderers = {
    overview: renderOverview,
    aspects: renderAspectAnalysis,
    sentiment: renderSentimentTrends,
    recommendations: renderRecommendations,
    feedback: renderFeedback,
  };

  return (
    <div className="dashboard-layout" id="dashboard-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar" id="dashboard-sidebar">
        <div className="sidebar-header">
          <div className="sidebar-avatar">🎓</div>
          <div className="sidebar-header-text">
            <h3>Faculty Dashboard</h3>
            <p>NLP Course · VI Sem AIML</p>
          </div>
        </div>

        <button className="sidebar-generate-btn" id="sidebar-generate-btn">
          + Generate Report
        </button>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <div
              key={item.id}
              className={`sidebar-nav-item ${activeView === item.id ? 'active' : ''}`}
              id={`sidebar-nav-${item.id}`}
              onClick={() => setActiveView(item.id)}
            >
              <span className="sidebar-nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-nav-item" onClick={onLogout}>
            <span className="sidebar-nav-icon">🚪</span>
            <span>Sign Out</span>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="dashboard-main">
        {/* Topbar */}
        <header className="dashboard-topbar" id="dashboard-topbar">
          <h1 className="topbar-title">Academic Insight Portal</h1>
          <div className="topbar-actions">
            <input
              type="text"
              className="topbar-search"
              placeholder="Search insights..."
              id="topbar-search"
            />
            <button className="topbar-icon-btn" title="Notifications">🔔</button>
            <button className="topbar-icon-btn" title="Settings">⚙️</button>
          </div>
        </header>

        {/* Content */}
        <div className="dashboard-content" id="dashboard-content">
          <span className="course-label">COURSE: NLP · VI SEMESTER · {totalResponses} RESPONSES</span>
          <div className="page-title-row">
            <h2 className="page-title">
              {activeView === 'overview' ? 'Aspect Performance Overview'
                : activeView === 'aspects' ? 'Aspect Analysis'
                : activeView === 'sentiment' ? 'Sentiment Trends'
                : activeView === 'recommendations' ? 'Pedagogical Recommendations'
                : 'Student Feedback'}
            </h2>
            <div className="page-title-actions">
              <button className="action-btn">📅 2024-25</button>
              <button className="action-btn primary">📄 Export PDF</button>
            </div>
          </div>

          {viewRenderers[activeView]?.() || renderOverview()}
        </div>
      </div>
    </div>
  );
}
