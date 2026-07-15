import os
import validators
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "text_summarizing"

app = Flask(__name__)

MAP_PROMPT_TEMPLATE = """
Extract the key points and main information from the following section of content.
Be concise and factual — this will later be combined with summaries of other sections.
content : {content}
"""

COMBINED_PROMPT_TEMPLATE = """
The following are summaries of different sections of the same piece of content:

{content}

Combine these into a single, coherent summary of approximately 300 words.
Then translate the final summary into {language}.
Return only the translated summary, with no additional commentary.
"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}

    api_key = (data.get("api_key") or "").strip()
    generic_url = (data.get("url") or "").strip()
    language = data.get("language") or "English"
    model_name = data.get("model") or "claude-fable-5"

    # ---- Same validation as the Streamlit version ----
    if not api_key or not generic_url:
        return jsonify({"error": "Please provide the important information"}), 400

    if not validators.url(generic_url):
        return jsonify({"error": "Please enter a valid Url. It may be a YT video url or website url"}), 400

    try:
        llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.gapgpt.app/v1",
            api_key=api_key,
        )

        # ---- Load content: YouTube vs website ----
        if "youtube.com" in generic_url:
            try:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                docs = loader.load()
            except Exception:
                return jsonify({"error": "This video has no available transcript."}), 400
        else:
            loader = UnstructuredURLLoader(
                urls=[generic_url],
                ssl_verify=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                },
            )
            docs = loader.load()

        # ---- Sanity check before running the chain ----
        if not docs or all(not doc.page_content.strip() for doc in docs):
            return jsonify({
                "error": "Couldn't extract any readable content from that URL. "
                         "The page may block scraping, or the video may have no transcript."
            }), 400

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        docs = splitter.split_documents(docs)

        map_prompt = PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["content"])
        combined_prompt = PromptTemplate(
            template=COMBINED_PROMPT_TEMPLATE, input_variables=["content", "language"]
        )

        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            combine_prompt=combined_prompt,
            map_prompt=map_prompt,
            combine_document_variable_name="content",
            map_reduce_document_variable_name="content",
        )

        output_summary = chain.invoke({"input_documents": docs, "language": language})

        parser = StrOutputParser()
        summary_text = parser.invoke(output_summary["output_text"])

        return jsonify({"summary": summary_text})

    except Exception as e:
        return jsonify({"error": f"Exception: {e}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(debug=True, host="0.0.0.0", port=port)
