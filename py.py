import validators
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  
os.environ["LANGCHAIN_PROJECT"] = "text_summarizing"

st.set_page_config(page_title="Text Summarizing with Langchain")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    openai_api_key = st.sidebar.text_input("Gapgpt API Key", type="password")
    language = st.sidebar.selectbox(
        "Translate to",
        options=["English", "French", "Persian", "Chinese", "Arabic"],
        index=0
    )
    model = st.sidebar.selectbox(
        "choose the model",
        options=["claude-fable-5", "gemini-3.5-flash", "grok-4.3", "deepseek-v4-pro", "o3-mini"],
        index=0
    )
generic_url = st.text_input(
    label="Please insert your website or youtube link which you want summarized content from",
    label_visibility="collapsed"
)

prompt_template = """
please provide me a summary of following content in 300 words : \n\n
{content}
and the translate that text into this language : {language}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["content","language"])

if st.button("summarize the content from YT or Website"):
    if not openai_api_key.strip() or not generic_url.strip():
        st.error("Please provide the important information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It may be a YT video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
               
                model = ChatOpenAI(
                    model=model,
                    base_url="https://api.gapgpt.app/v1",
                    api_key=openai_api_key
                )

                if "youtube.com" in generic_url.strip():
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url.strip(), add_video_info=True)
                        docs = loader.load()
                    except Exception:
                        st.error("This video has no available transcript.")
                        st.stop()               
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url.strip()],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
                docs = splitter.split_documents(docs)
                for doc in docs : 
                    print("*****************")
                    print(doc)
                    print("*****************")
                map_prompt_template = """
                    Extract the key points and main information from the following section of content.
                    Be concise and factual — this will later be combined with summaries of other sections.
                    content : {content}
                    """
                map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["content"])
                combined_prompt_template = """
                    The following are summaries of different sections of the same piece of content:

                    {content}

                    Combine these into a single, coherent summary of approximately 300 words.
                    Then translate the final summary into {language}.
                    Return only the translated summary, with no additional commentary.
                """ 
                combined_prompt = PromptTemplate(template=combined_prompt_template,input_variables=["content","language"])

                chain = load_summarize_chain(model, chain_type="map_reduce",combine_prompt = combined_prompt, map_prompt=map_prompt,combine_document_variable_name="content",map_reduce_document_variable_name="content")

                callback_handler = StreamlitCallbackHandler(st.container())
                output_summary = chain.invoke({"input_documents": docs, "language": language},config={"callbacks":[callback_handler]})  
                parser = StrOutputParser()
                st.success(parser.invoke(output_summary["output_text"]))
        except Exception as e:
            st.error(f"Exception: {e}")