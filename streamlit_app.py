import streamlit as st
import pandas as pd
import os
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
import openai
from openai import OpenAI


openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("📄 Invoice Data Extractor")

st.write("Upload one or more **Hungarian invoices (PDFs)** to extract relevant information.")

# Drag & Drop File Uploader
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Button to process files
if st.button("Extract Data"):
    if uploaded_files:
        extracted_data = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            pdf_content = ""

            # Try extracting text from the PDF
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    pdf_content += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
                continue

            # If text extraction fails, use OCR
            if len(pdf_content.strip()) < 100:
                pdf_content = ""
                try:
                    images = convert_from_bytes(uploaded_file.read())
                    for img in images:
                        pdf_content += pytesseract.image_to_string(img, lang="hun")
                except Exception as e:
                    st.error(f"OCR failed for {file_name}: {e}")
                    continue

            # OpenAI GPT prompt
            gpt_prompt = ("""I send you an extract of a pdf bill invoice in hungarian. Your job is to find the final several data from the invoice: """ 
                        + pdf_content +  """. Output the following in order: 
                        1) the name of the partner, 
                        2) the invoice number, 
                        3) the total gross amount of the full invoice, 
                        4) the total net amount of the invoice, 
                        5) the total VAT (ÁFA in hungarian) of the invoice. 
                        Output these values (1 and 2 as strings, 3, 4 and 5 as integers) separated by ; and nothing else!""")

            try:    
                client = OpenAI(api_key=openai.api_key)
    
                response = client.chat.completions.create(
                    model='gpt-4o', 
                    messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": gpt_prompt}],
                    max_tokens = 50,
                    temperature=0,
                    timeout=30)
                
                extracted_text = response.choices[0].message.content.strip()
    
                extracted_data.append([file_name] + extracted_text.split(";"))

            except Exception as e:
                st.error(f"GPT-4 extraction failed for {file_name}: {e}")
                continue
        
        if extracted_data:
            df = pd.DataFrame(extracted_data, columns=["File", "Partner", "Invoice Number", "Gross Amount", "Net Amount", "VAT"])
            st.write("✅ **Extraction complete!** Here are the results:")
            st.dataframe(df)

            # Offer CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "invoice_data.csv", "text/csv", key="download-csv")

    else:
        st.warning("⚠️ Please upload at least one PDF file.")


