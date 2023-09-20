import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import openai
import tempfile
import json

import spacy

def identify_missing_entities(article_content, included_entities):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article_content)
    new_entities = [ent.text for ent in doc.ents if ent.text not in included_entities]
    missing_entities = new_entities[:3]  # Limit to 1-3 entities
    return missing_entities

def generate_new_summary(current_summary, missing_entities):
    # Construct the prompt
    prompt = {
        "role": "user",
        "content": f"Current Summary: {current_summary}\nMissing Entities: {missing_entities}\nGenerate a new summary that includes these entities while maintaining the same word count."
    }

    # Send a request to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            prompt
        ]
    )

    # Extract the new summary from the response
    new_summary = response['choices'][0]['message']['content'].strip()

    return new_summary

# Initialize Streamlit
st.title("CoD Strategy for Article Summarization")

# Load environment variables
load_dotenv()

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# File Upload
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

# CoD Strategy Function
# CoD Strategy Function
def cod_strategy(retriever, llm, prompt_template):
    summaries = []
    # Initialize variables
    current_summary = ""
    included_entities = []

    # Retrieve relevant chunks from the article
    retrieved_docs = retriever.get_relevant_documents("Article Summary")
    article_content = " ".join([doc.page_content for doc in retrieved_docs])

    for i in range(5):
        # Identify missing entities from the article
        missing_entities = identify_missing_entities(article_content, included_entities)
        
        # Generate a new, denser summary
        new_summary = generate_new_summary(current_summary, missing_entities)
        
        # Update the current summary and included entities
        current_summary = new_summary
        included_entities.extend(missing_entities)
        
        # Append to summaries list
        summaries.append({
            "Missing_Entities": missing_entities,
            "Denser_Summary": new_summary
        })
    
    return summaries

# Main Logic
if uploaded_file is not None:
    st.write("File successfully uploaded. Press the button to start the CoD strategy.")
    
    if st.button("Start CoD"):
        st.write("Loading the PDF document...")
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
            temp_file_path = fp.name

        # Load the PDF document using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        
        # Split the text into chunks using TokenTextSplitter
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)

        # Generate embeddings using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Create a retriever using Chroma and the generated embeddings
        retriever = Chroma.from_documents(chunks, embeddings).as_retriever()

        # Extract the text from the PDF and replace {{ ARTICLE }} in the prompt
        context = " ".join([doc.page_content for doc in data])
        prompt_template = f"""
        Article: {context}
        You will generate increasingly concise, entity-dense summaries of the above article. 

        Repeat the following 2 steps 5 times. 

        Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary. 
        Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 

        A missing entity is:
        - relevant to the main story, 
        - specific yet concise (5 words or fewer), 
        - novel (not in the previous summary), 
        - faithful (present in the article), 
        - anywhere (can be located anywhere in the article).

        Guidelines:

        - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
        - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
        - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
        - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 

        Remember, use the exact same number of words for each summary.

        Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary". 
        """

        # Initialize the OpenAI API
        # Initialize the OpenAI API
        llm = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template}
            ]
        )

        # Call the CoD strategy function
        cod_summaries = cod_strategy(retriever, llm, prompt_template)   

        # Display the CoD summaries
        st.json(cod_summaries)
