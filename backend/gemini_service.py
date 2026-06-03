"""Gemini AI summary generation for pedagogy analytics."""
import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Cache: pedagogy_id -> (summary_text, response_count_at_generation)
_summary_cache: Dict[str, tuple] = {}

def _get_client():
    """Lazy-load Gemini client."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

def generate_pedagogy_summary(
    pedagogy_name: str,
    pedagogy_description: str,
    feedback_list: List[str],
    avg_ratings: Dict[str, float],
    sentiment_distribution: Dict[str, int],
    top_aspects: List[Dict],
    response_count: int,
) -> str:
    """Generate a structured AI summary for a pedagogy using Gemini."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Gemini API key not configured. Please add your GEMINI_API_KEY to the .env file."

    # Check cache
    cache_key = pedagogy_name
    if cache_key in _summary_cache:
        cached_summary, cached_count = _summary_cache[cache_key]
        if cached_count == response_count:
            return cached_summary

    # Build prompt
    feedback_block = "\n".join([f"  - \"{fb}\"" for fb in feedback_list[:30]])
    aspects_block = ", ".join([f"{a['aspect']} ({a['count']} mentions)" for a in top_aspects[:10]])

    prompt = f"""You are an expert educational consultant analyzing student feedback for a university course at RV College of Engineering, Bengaluru (VI Semester, AIML Department).

Analyze the following student feedback data for the teaching method: **{pedagogy_name}**
Description: {pedagogy_description}

### Quantitative Data (Average Ratings out of 5):
- Effectiveness: {avg_ratings.get('avg_effectiveness', 'N/A')}/5
- Engagement: {avg_ratings.get('avg_engagement', 'N/A')}/5  
- Clarity: {avg_ratings.get('avg_clarity', 'N/A')}/5
- Relevance: {avg_ratings.get('avg_relevance', 'N/A')}/5

### Sentiment Distribution:
- Positive aspects: {sentiment_distribution.get('Positive', 0)}
- Negative aspects: {sentiment_distribution.get('Negative', 0)}
- Neutral aspects: {sentiment_distribution.get('Neutral', 0)}

### Most Discussed Aspects:
{aspects_block}

### Raw Student Feedback ({response_count} responses, may include Hinglish/Hindi):
{feedback_block}

---

Provide a concise 3-paragraph analytical summary:

**Paragraph 1 - Overall Assessment:** Summarize the overall student perception of {pedagogy_name}. Mention the average ratings and whether students generally found this method effective.

**Paragraph 2 - Key Strengths:** Identify 2-3 specific strengths mentioned by students. Include brief direct quotes from feedback where impactful. If feedback is in Hinglish, translate the key sentiment.

**Paragraph 3 - Recommendations for Improvement:** Based on the negative feedback and lower-rated dimensions, suggest 2-3 specific, actionable improvements the instructor could make.

Keep the tone professional but constructive. Total length: 150-250 words."""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = response.text.strip()
        # Cache it
        _summary_cache[cache_key] = (summary, response_count)
        return summary
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return f"Failed to generate summary: {str(e)}"

def invalidate_cache(pedagogy_name: Optional[str] = None):
    """Invalidate cached summaries."""
    if pedagogy_name:
        _summary_cache.pop(pedagogy_name, None)
    else:
        _summary_cache.clear()
