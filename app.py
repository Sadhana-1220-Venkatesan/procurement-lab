import streamlit as st
import os
import openai
import re
import pandas as pd
import tempfile
import fitz
import docx
import pathlib
import pyocr.builders
import pytesseract
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply Gaussian blur for denoising
    denoised = cv2.GaussianBlur(thresholded, (3, 3), 0)
    
    return denoised

def get_pdf_text(pdf_docs):
    data={}
    document_name=""
    text = ""
    docu_text={}
    for pdf in pdf_docs:
        file_form=pathlib.Path(pdf).suffix
        document_name=os.path.basename(pdf)
        if file_form=='.doc' or file_form=='.docx':
            doc=docx.Document(pdf)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            text+='\n'.join(fullText)
            docu_text[document_name]=fullText
            if "dataframes" not in st.session_state:
                st.session_state.dataframes = {}
            data.update(docu_text)
        else:
            pdf_text = ""
            images = []
            fullText = []
            pdf_document = fitz.open(pdf, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pdf_text += page.get_text()


                img = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Adjust matrix as needed
                img_pil = Image.frombytes("RGB", [img.width, img.height], img.samples)
                images.append(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

            # OCR processing using pytesseract with image preprocessing
            ocr_results = []
            for img in images:
                img_preprocessed = preprocess_image(img)
                
                img_pil = Image.fromarray(img_preprocessed)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                    img_pil.save(temp_img, format="JPEG")
                    temp_img_path = temp_img.name
                    img_text = pytesseract.image_to_string(Image.open(temp_img_path))
                    ocr_results.append(img_text)
                    # Close the file before removing it
                    temp_img.close()
                    os.remove(temp_img_path)  # Delete the temporary file
            # Display content
                text += "\n".join([pdf_text] + ocr_results)
                fullText = "\n".join([pdf_text] + ocr_results)
                docu_text[document_name]=fullText
                if "dataframes" not in st.session_state:
                    st.session_state.dataframes = {}
                data.update(docu_text)

    st.session_state.dataframes=data
    print(data)
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    # HuggingFace Model

    #llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b-instruct", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def export_all_to_excel(dataframes):
    docs_directory = ''
    document_kv_pairs = {}
    document_texts={}
    for doc_file_name,document_text in dataframes.items():

        # Split the document into chunks and generate key-value pairs
        chunk_size = 1500
        document_chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]
        
        all_kv_pairs = []
        
        for chunk in document_chunks:
            prompt = f"Label the document chunk with key-value pairs where the key is an indicator and the value is the context mentioning the indicator. Indicators include 'effective date', 'contract type', 'master contract', 'start date', and more. If the chunk mentions any of these indicators, provide the context; otherwise, label it as 'not mentioned'.\n\nDocument Chunk:\n{chunk}\n\nKey-Value Pairs:\n"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
            )
            kv_pairs = response.choices[0].text.strip().split('\n')
            all_kv_pairs.extend(kv_pairs)
            parent=all_kv_pairs
        # Remove 'not mentioned' pairs from the list
        valid_kv_pairs = [pair for pair in all_kv_pairs if "not mentioned" not in pair.lower()]
        document_kv_pairs[doc_file_name] = valid_kv_pairs
    # Split the document names and store them in a list
        # Create a DataFrame from the data
        parsed_data = {}

# Process each document's data
        for document_name, values in document_kv_pairs.items():
            parsed_entries = {}
            for value in values:
                match = re.match(r'([^:]+):\s*(.*)', value)
                if match:
                    key, content = match.groups()
                    parsed_entries[key] = content
            parsed_data[document_name] = parsed_entries

        # Create a DataFrame from the parsed data
        df = pd.DataFrame(parsed_data)

        # Transpose the DataFrame for the desired layout
        result_df = df

        # Reset the index name and display the resulting DataFrame
        result_df.index.name = 'Document Name'
        output_file = "output.xlsx"
        result_df.to_excel(output_file)
    return result_df
def write(result):
    st.write(result)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "result_df" not in st.session_state:
        st.session_state.result_df = ""

    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    #st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your documents")
        # Set your default folder here
        default_folder = "C:/Users/vCreatek/A&D/DELL/"

        # Ensure the entered folder exists
        if not os.path.exists(default_folder):
            st.warning("Folder does not exist.")
        else:

            def list_files_and_folders(folder_path):
                items = os.listdir(folder_path)
                file_list = []
                folder_list = []

                for item in items:
                    item_path = os.path.join(folder_path, item)
                    if os.path.isdir(item_path):
                        folder_list.append(item_path)
                    else:
                        file_list.append(item_path)

                return file_list, folder_list

            # Store selected folders
            selected_folders = set()
            selected_files=[]
            def display_folder_contents(folder_path, level=1):
                files, subfolders = list_files_and_folders(folder_path)

                folder_checkbox_label = "" * (level-5) + f"📁 {os.path.basename(folder_path)}"
                folder_checkbox = st.checkbox(folder_checkbox_label, key=folder_path)

                # If the folder checkbox is selected, select all files in the folder
                if folder_checkbox:
                    selected_files.extend(files)

                if files:
                    for file in files:
                        checkbox_label = "" * (level-3) + os.path.basename(file)
                        file_selected = file in selected_files
                        # Check the file if its folder is in the set of selected folders
                        file_checkbox=st.checkbox(checkbox_label, value=file_selected)
                        if file_checkbox and not file_selected:
                            selected_files.append(file)
                        elif not file_checkbox and file_selected:
                            selected_files.remove(file)

                for subfolder in subfolders:
                    display_folder_contents(subfolder, level + 1)
                return selected_files

            files=display_folder_contents(default_folder)


        if st.button("Process"):
            with  st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(files)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.button("Export DataFrames to Excel"):
            st.session_state.result_df=export_all_to_excel(st.session_state.dataframes)
    write(st.session_state.result_df)




if __name__ == '__main__':
    main()