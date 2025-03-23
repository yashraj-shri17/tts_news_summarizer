import os
import time
import logging
import requests
import json
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline, MarianMTModel, MarianTokenizer
from rake_nltk import Rake
import nltk
from gtts import gTTS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sentiment_analyzer = SentimentIntensityAnalyzer()
r = Rake()

NEWS_API_KEY = "70d59b5e128141d68b5789136b6af881"

model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def translate_to_hindi(text):
    try:
        text = text[:400]
        batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated = model.generate(**batch)
        hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return hindi_text
    except:
        return text

def fetch_news_from_newsapi(company_name, num_articles=10):
    url = f"https://newsapi.org/v2/everything?q={company_name}&pageSize={num_articles}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    articles = response.json().get('articles', [])
    article_data_list = []

    for article in articles:
        try:
            art = Article(article['url'])
            art.download()
            art.parse()

            text = art.text[:500]
            score = sentiment_analyzer.polarity_scores(text)
            compound = score['compound']
            sentiment = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"

            classification = classifier(text, candidate_labels=["investment opportunity", "risk factors", "market analysis", "company growth", "financial report"])
            categories = classification['labels'][:3]

            r.extract_keywords_from_text(art.text)
            topics = r.get_ranked_phrases()[:5]

            article_data_list.append({
                "Title": article['title'],
                "Summary": article['description'] or art.text[:500],
                "Sentiment": sentiment,
                "Topics": topics,
                "Categories": categories
            })
            time.sleep(1)
        except:
            pass

    return article_data_list

def comparative_sentiment_analysis(articles, company_name):
    sentiment_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for article in articles:
        sentiment_count[article["Sentiment"]] += 1

    all_topics = [set(a["Topics"]) for a in articles]
    common_topics = set.intersection(*all_topics) if len(all_topics) > 1 else list(all_topics[0])
    unique_topics = [list(topics - common_topics) for topics in all_topics]

    coverage_differences = []
    for i in range(len(articles)-1):
        comparison = {
            "Comparison": f"Article {i+1} talks about {articles[i]['Summary'][:50]}..., while Article {i+2} focuses on {articles[i+1]['Summary'][:50]}...",
            "Impact": "Investors may interpret differently based on focus areas."
        }
        coverage_differences.append(comparison)

    overall_sentiment = "mostly positive" if sentiment_count["Positive"] > sentiment_count["Negative"] else "mixed or cautious"
    final_conclusion = f"{company_name}’s latest news coverage seems {overall_sentiment}."

    return {
        "Sentiment Distribution": sentiment_count,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": {
            "Common Topics": list(common_topics),
            "Unique Topics per Article": unique_topics
        },
        "Final Sentiment Analysis": final_conclusion
    }

def convert_combined_summaries_to_hindi_audio(articles, audio_file_name="combined_summary_hindi.mp3"):
    if not articles:
        return ""

    combined_summary = " ".join([article["Summary"] for article in articles if article.get("Summary")])
    if not combined_summary.strip():
        return ""
    try:
        if not os.path.exists("audio_files"):
            os.makedirs("audio_files")

        hindi_summary = translate_to_hindi(combined_summary)
        audio_path = os.path.join("audio_files", audio_file_name)
        tts = gTTS(text=hindi_summary, lang='hi')
        tts.save(audio_path)
        return audio_path
    except:
        return ""

if __name__ == "__main__":
    company_name = "Meta"
    articles = fetch_news_from_newsapi(company_name, num_articles=10)
    comparative_data = comparative_sentiment_analysis(articles, company_name) if articles else {}
    audio_path = convert_combined_summaries_to_hindi_audio(articles, "combined_summary_hindi.mp3")

    final_output = {
        "Company": company_name,
        "Articles": articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": comparative_data.get("Sentiment Distribution", {}),
            "Coverage Differences": comparative_data.get("Coverage Differences", []),
            "Topic Overlap": comparative_data.get("Topic Overlap", {})
        },
        "Final Sentiment Analysis": comparative_data.get("Final Sentiment Analysis", ""),
        "Audio": f"[Play Hindi Speech] - Combined summaries audio saved at {audio_path}"
    }

    print(json.dumps(final_output, indent=4, ensure_ascii=False))
