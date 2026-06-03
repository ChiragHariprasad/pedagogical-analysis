import { useState, useEffect, useCallback } from 'react';
import Plot from './PlotWrapper';

const PEDAGOGY_META = {
  traditional_lecture: { name: 'Traditional Lecture', emoji: '🎓', color: 'hsl(220, 70%, 60%)', rgb: '76, 114, 204' },
  project_based: { name: 'Project-Based Learning', emoji: '🛠️', color: 'hsl(160, 70%, 50%)', rgb: '38, 194, 129' },
  flipped_classroom: { name: 'Flipped Classroom', emoji: '🔄', color: 'hsl(280, 70%, 60%)', rgb: '163, 82, 214' },
  collaborative: { name: 'Collaborative Learning', emoji: '👥', color: 'hsl(30, 80%, 55%)', rgb: '224, 145, 45' },
  inquiry_based: { name: 'Inquiry-Based Learning', emoji: '🔍', color: 'hsl(340, 70%, 60%)', rgb: '214, 71, 106' },
  experiential_labs: { name: 'Experiential Labs', emoji: '🧪', color: 'hsl(50, 80%, 50%)', rgb: '214, 190, 25' },
};

const DIMENSIONS = ['effectiveness', 'engagement', 'clarity', 'relevance'];
const DIM_LABELS = ['Effectiveness', 'Engagement', 'Clarity', 'Relevance'];
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function CountingNumber({ target, duration = 1500 }) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (target === 0) { setValue(0); return; }
    const start = 0;
    const startTime = performance.now();

    const step = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.floor(start + (target - start) * eased));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration]);

  return <span className="counting-animation">{value}</span>;
}

export default function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [responses, setResponses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedCard, setExpandedCard] = useState(null);
  const [geminiSummaries, setGeminiSummaries] = useState({});
  const [geminiLoading, setGeminiLoading] = useState({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [analyticsRes, responsesRes] = await Promise.all([
        fetch(`${API_BASE}/api/analytics`),
        fetch(`${API_BASE}/api/responses`),
      ]);

      if (!analyticsRes.ok) throw new Error('Failed to fetch analytics');

      const analyticsData = await analyticsRes.json();
      setAnalytics(analyticsData);

      if (responsesRes.ok) {
        const responsesData = await responsesRes.json();
        setResponses(Array.isArray(responsesData) ? responsesData : responsesData.responses || []);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const fetchGeminiSummary = useCallback(async (pedagogyId) => {
    if (geminiSummaries[pedagogyId] || geminiLoading[pedagogyId]) return;
    setGeminiLoading((prev) => ({ ...prev, [pedagogyId]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/analytics/${pedagogyId}`);
      if (res.ok) {
        const data = await res.json();
        setGeminiSummaries((prev) => ({ ...prev, [pedagogyId]: data.summary || data.gemini_summary || 'No summary available.' }));
      } else {
        setGeminiSummaries((prev) => ({ ...prev, [pedagogyId]: 'Summary not available.' }));
      }
    } catch {
      setGeminiSummaries((prev) => ({ ...prev, [pedagogyId]: 'Failed to load summary.' }));
    } finally {
      setGeminiLoading((prev) => ({ ...prev, [pedagogyId]: false }));
    }
  }, [geminiSummaries, geminiLoading]);

  const handleCardExpand = (pedagogyId) => {
    const newExpanded = expandedCard === pedagogyId ? null : pedagogyId;
    setExpandedCard(newExpanded);
    if (newExpanded) fetchGeminiSummary(newExpanded);
  };

  if (loading) {
    return (
      <div className="dashboard-container" id="dashboard-container">
        <div className="dashboard-loading" id="dashboard-loading">
          <div className="dashboard-loading-spinner" />
          <p className="dashboard-loading-text">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container" id="dashboard-container">
        <div className="dashboard-empty" id="dashboard-error">
          <div className="dashboard-empty-icon">⚠️</div>
          <h3>Could not load dashboard</h3>
          <p>{error}. Make sure the backend server is running at {API_BASE}.</p>
          <button className="btn-primary" id="dashboard-retry-btn" onClick={fetchData}>
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  /* ── Normalize backend response into { [pedagogy_id]: stats } object ── */
  // Backend returns: { total_responses, total_surveys, pedagogy_analytics: [...] }
  const rawPedagogyList = analytics.pedagogy_analytics || analytics.pedagogies || [];
  const isArray = Array.isArray(rawPedagogyList);

  if (!analytics || (isArray && rawPedagogyList.length === 0) || (!isArray && Object.keys(rawPedagogyList).length === 0)) {
    return (
      <div className="dashboard-container" id="dashboard-container">
        <div className="dashboard-empty" id="dashboard-empty">
          <div className="dashboard-empty-icon">📊</div>
          <h3>No Data Yet</h3>
          <p>No survey responses have been submitted. Complete the survey to see analytics here.</p>
        </div>
      </div>
    );
  }

  /* Convert array → object keyed by pedagogy_id */
  const pedagogyStats = isArray
    ? rawPedagogyList.reduce((acc, item) => { acc[item.pedagogy_id] = item; return acc; }, {})
    : rawPedagogyList;
  const totalResponses = analytics.total_responses || 0;

  /* Build global top_aspects from per-pedagogy data */
  const globalAspectMap = {};
  Object.values(pedagogyStats).forEach((stats) => {
    (stats.top_aspects || []).forEach((a) => {
      const name = a.aspect || a.name || 'unknown';
      if (!globalAspectMap[name]) globalAspectMap[name] = { aspect: name, count: 0, sentiment: 'neutral' };
      globalAspectMap[name].count += a.count || 0;
    });
  });
  const globalTopAspects = Object.values(globalAspectMap).sort((a, b) => b.count - a.count).slice(0, 15);

  /* ── Radar Chart Data ── */
  const radarTraces = Object.entries(pedagogyStats).map(([pid, stats]) => {
    const meta = PEDAGOGY_META[pid] || { name: pid, color: '#888', rgb: '136,136,136' };
    const avgRatings = DIMENSIONS.map((d) => {
      if (stats.avg_ratings) return stats.avg_ratings[d] || 0;
      if (stats.averages) return stats.averages[d] || 0;
      return stats[`avg_${d}`] || 0;
    });
    return {
      type: 'scatterpolar',
      r: [...avgRatings, avgRatings[0]], // close the polygon
      theta: [...DIM_LABELS, DIM_LABELS[0]],
      fill: 'toself',
      fillcolor: `rgba(${meta.rgb}, 0.1)`,
      line: { color: meta.color, width: 2 },
      name: meta.name,
      hoverinfo: 'name+r+theta',
    };
  });

  const radarLayout = {
    polar: {
      bgcolor: 'transparent',
      radialaxis: {
        visible: true,
        range: [0, 5],
        tickvals: [1, 2, 3, 4, 5],
        tickfont: { color: '#64748b', size: 10 },
        gridcolor: 'rgba(255,255,255,0.06)',
        linecolor: 'transparent',
      },
      angularaxis: {
        tickfont: { color: '#94a3b8', size: 11 },
        gridcolor: 'rgba(255,255,255,0.06)',
        linecolor: 'rgba(255,255,255,0.08)',
      },
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9' },
    showlegend: true,
    legend: { font: { color: '#94a3b8', size: 11 }, bgcolor: 'transparent', x: 1.05, y: 1 },
    margin: { t: 40, b: 40, l: 60, r: 120 },
    autosize: true,
  };

  /* ── Sentiment Bar Chart ── */
  const sentimentPedagogies = Object.keys(pedagogyStats);
  const positiveData = sentimentPedagogies.map((pid) => {
    const s = pedagogyStats[pid];
    return (s.sentiment_distribution?.Positive || s.sentiment?.positive || s.positive_count || 0);
  });
  const negativeData = sentimentPedagogies.map((pid) => {
    const s = pedagogyStats[pid];
    return (s.sentiment_distribution?.Negative || s.sentiment?.negative || s.negative_count || 0);
  });

  const sentimentTraces = [
    {
      x: sentimentPedagogies.map((pid) => (PEDAGOGY_META[pid]?.name || pid).replace(/ /g, '<br>')),
      y: positiveData,
      name: 'Positive',
      type: 'bar',
      marker: { color: '#10b981', borderRadius: 4 },
    },
    {
      x: sentimentPedagogies.map((pid) => (PEDAGOGY_META[pid]?.name || pid).replace(/ /g, '<br>')),
      y: negativeData,
      name: 'Negative',
      type: 'bar',
      marker: { color: '#ef4444', borderRadius: 4 },
    },
  ];

  const sentimentLayout = {
    barmode: 'group',
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9', size: 11 },
    xaxis: { tickfont: { color: '#94a3b8', size: 9 }, gridcolor: 'rgba(255,255,255,0.04)' },
    yaxis: { title: 'Aspect Count', titlefont: { color: '#94a3b8' }, tickfont: { color: '#94a3b8' }, gridcolor: 'rgba(255,255,255,0.06)' },
    legend: { font: { color: '#94a3b8' }, bgcolor: 'transparent' },
    margin: { t: 20, b: 80, l: 50, r: 20 },
    autosize: true,
    bargap: 0.3,
    bargroupgap: 0.1,
  };

  /* ── Top Aspects Horizontal Bar ── */
  const topAspects = globalTopAspects;
  const aspectNames = topAspects.map((a) => a.aspect || a.name || 'Unknown');
  const aspectCounts = topAspects.map((a) => a.count || a.mentions || 0);
  const aspectColors = topAspects.map((a) => {
    const sentiment = a.sentiment || a.dominant_sentiment || 'neutral';
    if (sentiment.toLowerCase() === 'positive') return '#10b981';
    if (sentiment.toLowerCase() === 'negative') return '#ef4444';
    return '#64748b';
  });

  const aspectTrace = {
    y: aspectNames.reverse(),
    x: aspectCounts.reverse(),
    type: 'bar',
    orientation: 'h',
    marker: { color: aspectColors.reverse() },
    hoverinfo: 'x+y',
  };

  const aspectLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#f1f5f9', size: 11 },
    xaxis: { title: 'Mentions', titlefont: { color: '#94a3b8' }, tickfont: { color: '#94a3b8' }, gridcolor: 'rgba(255,255,255,0.06)' },
    yaxis: { tickfont: { color: '#94a3b8', size: 10 }, automargin: true },
    margin: { t: 10, b: 40, l: 140, r: 20 },
    autosize: true,
  };

  const plotConfig = { displayModeBar: false, responsive: true };

  /* ── Drill-Down Data ── */
  const getResponsesForPedagogy = (pid) => {
    if (!Array.isArray(responses)) return [];
    return responses.filter((r) => r.pedagogy_id === pid || r.pedagogy === pid);
  };

  const getAvgRating = (stats) => {
    const avgObj = stats.avg_ratings || stats.averages || {};
    const vals = DIMENSIONS.map((d) => avgObj[d] || stats[`avg_${d}`] || 0).filter((v) => v > 0);
    return vals.length > 0 ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1) : '—';
  };

  const getTopAspectsForPedagogy = (stats) => {
    const aspects = stats.top_aspects || stats.aspects || [];
    return aspects.slice(0, 3);
  };

  return (
    <div className="dashboard-container" id="dashboard-container">
      {/* Header */}
      <div className="dashboard-header" id="dashboard-header">
        <h1 className="dashboard-title gradient-text" id="dashboard-title">
          Pedagogical Intelligence Dashboard
        </h1>
        <div className="dashboard-response-count" id="dashboard-response-count">
          📋 <CountingNumber target={totalResponses} /> <span>total responses</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="dashboard-grid" id="dashboard-charts">
        {/* Radar Chart */}
        <div className="chart-card" id="chart-radar">
          <div className="chart-card-title">
            <span className="chart-icon">🕸️</span>
            Rating Dimensions — All Pedagogies
          </div>
          <Plot
            data={radarTraces}
            layout={radarLayout}
            config={plotConfig}
            style={{ width: '100%', height: '420px' }}
            useResizeHandler
          />
        </div>

        {/* Sentiment Distribution */}
        <div className="chart-card" id="chart-sentiment">
          <div className="chart-card-title">
            <span className="chart-icon">📊</span>
            Sentiment Distribution
          </div>
          <Plot
            data={sentimentTraces}
            layout={sentimentLayout}
            config={plotConfig}
            style={{ width: '100%', height: '420px' }}
            useResizeHandler
          />
        </div>

        {/* Top Aspects */}
        {topAspects.length > 0 && (
          <div className="chart-card dashboard-grid-full" id="chart-aspects">
            <div className="chart-card-title">
              <span className="chart-icon">🏷️</span>
              Top Mentioned Aspects
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 400, marginLeft: '0.5rem' }}>
                (Green = Positive, Red = Negative)
              </span>
            </div>
            <Plot
              data={[aspectTrace]}
              layout={aspectLayout}
              config={plotConfig}
              style={{ width: '100%', height: `${Math.max(300, topAspects.length * 28)}px` }}
              useResizeHandler
            />
          </div>
        )}
      </div>

      {/* Drill-Down Cards */}
      <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', marginBottom: '0.5rem', marginTop: '1rem' }} id="drilldown-title">
        Per-Pedagogy Insights
      </h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        Click any card to expand and see detailed feedback + AI-generated summary.
      </p>

      <div className="drilldown-grid" id="drilldown-grid">
        {Object.entries(pedagogyStats).map(([pid, stats]) => {
          const meta = PEDAGOGY_META[pid] || { name: pid, emoji: '📋', color: '#888' };
          const isExpanded = expandedCard === pid;
          const avg = getAvgRating(stats);
          const topAspList = getTopAspectsForPedagogy(stats);
          const pedResponses = getResponsesForPedagogy(pid);

          // Sentiment counts for mini donut
          const posCount = stats.sentiment_distribution?.Positive || stats.sentiment?.positive || stats.positive_count || 0;
          const negCount = stats.sentiment_distribution?.Negative || stats.sentiment?.negative || stats.negative_count || 0;
          const totalSent = posCount + negCount;

          const donutTrace = {
            values: totalSent > 0 ? [posCount, negCount] : [1],
            labels: totalSent > 0 ? ['Positive', 'Negative'] : ['No data'],
            type: 'pie',
            hole: 0.65,
            marker: {
              colors: totalSent > 0 ? ['#10b981', '#ef4444'] : ['#334155'],
            },
            textinfo: 'none',
            hoverinfo: totalSent > 0 ? 'label+percent' : 'none',
          };

          const donutLayout = {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            showlegend: false,
            margin: { t: 5, b: 5, l: 5, r: 5 },
            autosize: true,
            annotations: totalSent > 0
              ? [{
                  text: `${Math.round((posCount / totalSent) * 100)}%`,
                  font: { size: 13, color: '#f1f5f9', family: 'Inter' },
                  showarrow: false,
                }]
              : [],
          };

          return (
            <div
              key={pid}
              className={`drill-down-card ${isExpanded ? 'expanded' : ''}`}
              style={{ '--card-accent': meta.color }}
              id={`drilldown-card-${pid}`}
              onClick={() => handleCardExpand(pid)}
            >
              <div className="drill-down-card-header">
                <span className="drill-down-card-emoji">{meta.emoji}</span>
                <span className="drill-down-card-name">{meta.name}</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div className="drill-down-card-avg">
                  <span className="avg-value">{avg}</span>
                  <span className="avg-label">/ 5<br />avg rating</span>
                </div>
                <div style={{ width: '80px', height: '80px', flexShrink: 0 }}>
                  <Plot
                    data={[donutTrace]}
                    layout={donutLayout}
                    config={{ displayModeBar: false, responsive: true, staticPlot: true }}
                    style={{ width: '80px', height: '80px' }}
                  />
                </div>
              </div>

              {topAspList.length > 0 && (
                <div className="drill-down-aspects" id={`drilldown-aspects-${pid}`}>
                  {topAspList.map((a, i) => {
                    const sentiment = (a.sentiment || a.dominant_sentiment || 'neutral').toLowerCase();
                    return (
                      <span key={i} className={`drill-down-aspect-tag ${sentiment === 'positive' ? 'positive' : sentiment === 'negative' ? 'negative' : ''}`}>
                        {a.aspect || a.name}
                      </span>
                    );
                  })}
                </div>
              )}

              {isExpanded && (
                <div className="drill-down-expanded-content" id={`drilldown-expanded-${pid}`} onClick={(e) => e.stopPropagation()}>
                  {/* Gemini Summary */}
                  <div className="gemini-summary" id={`gemini-summary-${pid}`}>
                    <div className="gemini-summary-header">
                      <span className="sparkle-icon">✨</span>
                      <span>AI-Generated Summary</span>
                    </div>
                    {geminiLoading[pid] ? (
                      <div className="gemini-summary-loading">Generating summary</div>
                    ) : (
                      <p className="gemini-summary-text">
                        {geminiSummaries[pid] || 'Click to load summary...'}
                      </p>
                    )}
                  </div>

                  {/* Individual feedback */}
                  {pedResponses.length > 0 && (
                    <div style={{ marginTop: '1rem' }}>
                      <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        Individual Feedback ({pedResponses.length})
                      </h4>
                      {pedResponses.slice(0, 10).map((resp, i) => (
                        <div key={i} className="drill-down-feedback-item" id={`feedback-item-${pid}-${i}`}>
                          <div className="feedback-rating">
                            ★ {resp.effectiveness || resp.rating || '—'} eff. | ★ {resp.engagement || '—'} eng. | ★ {resp.clarity || '—'} clar. | ★ {resp.relevance || '—'} rel.
                          </div>
                          <div>{resp.feedback || resp.text || 'No written feedback'}</div>
                        </div>
                      ))}
                      {pedResponses.length > 10 && (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                          + {pedResponses.length - 10} more responses
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
