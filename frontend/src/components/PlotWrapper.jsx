import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

/**
 * Thin wrapper around plotly.js-dist-min.
 * Drop-in replacement for <Plot data={} layout={} config={} />.
 */
export default function Plot({ data = [], layout = {}, config = {}, style = {}, useResizeHandler = false, className = '' }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(ref.current, data, layout, config);
  }, [data, layout, config]);

  useEffect(() => {
    if (!useResizeHandler || !ref.current) return;
    const handleResize = () => Plotly.Plots.resize(ref.current);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [useResizeHandler]);

  useEffect(() => {
    return () => {
      if (ref.current) Plotly.purge(ref.current);
    };
  }, []);

  return <div ref={ref} style={style} className={className} />;
}
