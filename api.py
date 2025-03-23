from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils1 import fetch_news_from_newsapi, comparative_sentiment_analysis, convert_combined_summaries_to_hindi_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze/")
def analyze(company_name: str):
    articles = fetch_news_from_newsapi(company_name, num_articles=10)
    if not articles:
        return {
            "Company": company_name,
            "Articles": [],
            "Comparative Sentiment Score": {},
            "Coverage Differences": [],
            "Topic Overlap": {},
            "Final Sentiment Analysis": "No articles found for sentiment analysis.",
            "Audio_File": None
        }
    
    comparative_data = comparative_sentiment_analysis(articles, company_name)
    # Add fallback values in case keys are missing
    sentiment_score = comparative_data.get("Sentiment Distribution", {})
    coverage_differences = comparative_data.get("Coverage Differences", [])
    topic_overlap = comparative_data.get("Topic Overlap", {})
    final_sentiment = comparative_data.get("Final Sentiment Analysis", "Analysis not available.")

    
    audio_path = convert_combined_summaries_to_hindi_audio(articles, "combined_summary_hindi.mp3")
    
    return {
        "Company": company_name,
        "Articles": articles,
        "Comparative Sentiment Score": sentiment_score,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": topic_overlap,
        "Final Sentiment Analysis": final_sentiment,
        "Audio_File": audio_path
    }
