import streamlit as st
import requests
import gradio as gr
import json

st.title("📈 Company News & Sentiment Analysis Tool")
company_name = st.text_input("Enter Company Name", value="Meta")

if st.button("Fetch News & Analyze"):
    with st.spinner("Fetching and Analyzing..."):
        try:
            response = requests.get(f"http://localhost:8000/analyze/?company_name={company_name}")
            data = response.json()

            if not data["Articles"]:
                st.warning("No articles found for the given company.")
            else:
                st.subheader(f"News Articles for {data['Company']}")
                for i, article in enumerate(data["Articles"]):
                    st.markdown(f"### Article {i+1}: {article['Title']}")
                    st.write(f"**Summary:** {article['Summary']}")
                    st.write(f"**Sentiment:** {article['Sentiment']}")
                    st.write(f"**Topics:** {', '.join(article['Topics'])}")
                    st.write("---")

                st.subheader("Comparative Sentiment Analysis")
                st.json(data.get("Comparative Sentiment Score", {}))
                
                st.write("**Coverage Differences:**")
                coverage_diff = data.get("Coverage Differences", [])
                if coverage_diff:
                    for diff in coverage_diff:
                        st.write(f"- {diff['Comparison']}")
                        st.write(f"  Impact: {diff['Impact']}")
                else:
                    st.write("No coverage differences found.")

                st.subheader("Topic Overlap")
                st.json(data.get("Topic Overlap", {}))

                st.success(data.get("Final Sentiment Analysis", "No final sentiment available."))

                audio_file = data.get("Audio_File")
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                    st.write(f"Download Hindi audio summary: [{audio_file}]({audio_file})")

        except Exception as e:
            st.error(f"Error: {e}")

# Optional: Gradio UI
def gradio_ui(company_name):
    response = requests.get(f"http://localhost:8000/analyze/?company_name={company_name}")
    data = response.json()
    summary_text = ""
    for i, article in enumerate(data["Articles"]):
        summary_text += f"\nArticle {i+1}: {article['Title']}\nSummary: {article['Summary']}\nSentiment: {article['Sentiment']}\n\n"
    summary_text += "\nComparative Analysis:\n" + json.dumps(data.get("Comparative Sentiment Score", {}), indent=2)
    audio_link = data.get("Audio_File")
    return summary_text, audio_link

with st.expander("Gradio UI"):
    demo = gr.Interface(fn=gradio_ui,
                        inputs="text",
                        outputs=["text", "audio"],
                        title="Company News Analyzer (Gradio)")
    demo.render()
