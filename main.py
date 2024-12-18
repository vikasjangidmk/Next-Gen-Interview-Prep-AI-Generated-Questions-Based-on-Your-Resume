import streamlit as st
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

def process_resume_pdf(file):
    """Extract text from a PDF file."""
    if file:
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages)
        return text
    return ""

def preproces_job_posting(url, file):
    if file is None:
        raise ValueError("No file uploaded.")
    
    # Load job posting from URL
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content if url else ""

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
        df = pd.read_csv(file)
        skills = df['Technology'].tolist()
    elif file.name.endswith('.pdf'):
        resume_text = process_resume_pdf(file)
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
        You are a Robot.
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

def streamlit_interface():
    st.title("Job Scraping & Analyzer with Interview Preparation Questions Using Gen-AI")

    url_input = st.text_input("Website URL", placeholder="Enter the URL of the job posting")
    file_input = st.file_uploader("Upload Resume or Portfolio File (CSV or PDF)")

    if st.button("Analyze Job Posting and Resume"):
        if url_input and file_input:
            try:
                result = preproces_job_posting(url_input, file_input)
                st.json(result)
            except ValueError as ve:
                st.error(f"Error: {str(ve)}")
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")
        else:
            st.error("Please provide both the URL and file to analyze.")

# Run the app
streamlit_interface()
