import streamlit as st
import pandas as pd
import os
import PyPDF2
import pdf2image
import pytesseract
from io import BytesIO
import openai
from openai import OpenAI


openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]



st.title("📄 Invoice Data Extractor")

st.write("Upload one or more **Hungarian invoices (PDFs)** to extract relevant information.")

#0) Drag & Drop File Uploader
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)



#1) text extraction from pdf
extracted_text_from_invoice = []
if st.button("Extract Data"):
    if uploaded_files:
        if len(uploaded_files) > 50:
            st.write("Parsing the first 50 files.")
        for uploaded_file in uploaded_files[:50]:
            file_name = uploaded_file.name
            pdf_content = ""

            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    pdf_content += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading {file_name}: {e}")
                continue

            if len(pdf_content.strip()) < 100:
                pdf_content = ""
                try:
                    uploaded_file.seek(0)
                    images = pdf2image.convert_from_bytes(uploaded_file.read())
                    for img in images:
                        pdf_content += pytesseract.image_to_string(img, lang="hun")
                except Exception as e:
                    st.error(f"OCR failed for {file_name}: {e}")
                    continue
                
            if len(pdf_content) > 5000:
                st.write(file_name + " file is too big. Only parsing the first 5000 characters.")
                pdf_content = pdf_content[:5000]
            
            extracted_text_from_invoice.append([file_name, pdf_content])
    else:
        st.warning("⚠️ Please upload at least one PDF file.")


#2) data extraction from text
extracted_data = []
if extracted_text_from_invoice:    
    for i in range(len(extracted_text_from_invoice)):
        file_name = extracted_text_from_invoice[i][0]
        pdf_content = extracted_text_from_invoice[i][1]
        
        gpt_prompt = ("""I send you an extract of a pdf bill invoice in hungarian. Your job is to find the final several data from the invoice: """ 
                    + pdf_content +  """. Output the following in order: 
                    1) the name of the partner, 
                    2) the invoice number, 
                    3) the date of the invoice,
                    4) the total gross amount of the full invoice, 
                    5) the total net amount of the invoice, 
                    6) the total VAT (ÁFA in hungarian) of the invoice. 
                    Output these values (1, 2 and 3 as strings, 4, 5 and 6 as integers) separated by ; and nothing else!""")

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
            
            if len(extracted_text.split(";")) != 6:
                st.error(f"GPT-4 extraction failed for {file_name}: {e}")
                continue        
            extracted_data.append([file_name] + extracted_text.split(";"))
        
        except Exception as e:
            st.error(f"GPT-4 extraction failed for {file_name}: {e}")
            continue




#3) Drag & Drop File Uploader for excel
st.write("Upload the excel sheet to verify the results.")
uploaded_excel_file = st.file_uploader("Upload Excel file", type=["xlsx"], accept_multiple_files=False)  


#3) output csv
if len(extracted_data) != 0:
    if uploaded_excel_file:
        df_excel = pd.read_excel(uploaded_excel_file, sheet_name='Mintavétel')
        st.dataframe(df_excel.head())
        df = pd.DataFrame(extracted_data, columns=["File", "Partner", "Invoice Number", "Invoice Date", "Gross Amount", "Net Amount", "VAT"])
        st.write("✅ **Extraction complete!** Here are the results:")
        st.dataframe(df)
    
        # Offer CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "invoice_data.csv", "text/csv", key="download-csv")
