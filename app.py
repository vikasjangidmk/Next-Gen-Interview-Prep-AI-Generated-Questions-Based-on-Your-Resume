import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import os

# Set up the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

def process_resume_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text

def preproces_job_posting(url, file):
    # Load job posting from URL
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content

    # Extract job postings in JSON format
    prompt_extract = PromptTemplate.from_template("""
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
        following keys: `role`, `experience`, `skills`, and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):
        """)
    chain_extract = prompt_extract | llm
    res_1 = chain_extract.invoke(input={'page_data': page_data})
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res_1.content)

    # Process uploaded file (either CSV or PDF)
    if file.name.endswith('.csv'):
        df = pd.read_csv(file.name)
        skills = df['Technology'].tolist()
    elif file.name.endswith('.pdf'):
        resume_text = process_resume_pdf(file.name)
        skills = resume_text.split()  # Split PDF text into words for basic skill extraction
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or PDF file.")

    # Store skills in ChromaDB
    client = chromadb.PersistentClient('vectorstore')
    collections = client.get_or_create_collection(name="portfolio_app")
    if not collections.count():
        for skill in skills:
            collections.add(documents=skill, ids=[str(uuid.uuid4())])

    # Match job description with skills and generate questions
    job = json_res.get('skills', []) if type(json_res) == dict else json_res[0].get('skills', [])

    prompt_skills_and_question = PromptTemplate.from_template("""
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are Mishu Dhar Chando, the CEO of Knowledge Doctor, a YouTube channel specializing in educating individuals on machine learning, deep learning, and natural language processing.
        Your expertise lies in bridging the gap between theoretical knowledge and practical applications through engaging content and innovative problem-solving techniques.
        Your job is to:
        1. Analyze the given job description to identify the required technical skills and match them with the provided skill set to calculate a percentage match.
        2. Generate a list of 20-30 tailored interview questions based on the job description.
        3. Return the information in JSON format with the following keys:
            - `skills_match`: A dictionary where each key is a skill, and the value is the matching percentage.
            - `interview_questions`: A list of tailored questions related to the job description.

        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):

        """)
    chain_skills_and_question = prompt_skills_and_question | llm
    res1 = chain_skills_and_question.invoke({"job_description": str(job)})
    final_result = json_parser.parse(res1.content)
    return final_result

def gradio_interface(url, file):
    try:
        result = preproces_job_posting(url, file)
        return result
    except Exception as e:
        return {"error": str(e)}

# Gradio Interface
with gr.Blocks(theme='Respair/Shiki@1.2.1') as app:
    gr.Markdown("# Job Scraping & Analyzer with Interview Preparation Questions Using Gen-AI")

    with gr.Row():
        url_input = gr.Textbox(label="Website URL", placeholder="Enter the URL of the job posting")
        file_input = gr.File(label="Upload Resume or Portfolio File (CSV or PDF)")

    analyze_button = gr.Button("Analyze Job Posting and Resume")
    output_box = gr.JSON(label="Result")

    analyze_button.click(gradio_interface, inputs=[url_input, file_input], outputs=output_box)

app.launch()
