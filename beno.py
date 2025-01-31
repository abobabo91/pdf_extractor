import pandas as pd
import os
import PyPDF2

from pdf2image import convert_from_path
import pytesseract

import openai
from openai import OpenAI

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = "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"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update with your path

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


def extract_text_via_ocr(pdf_path):
    """Extracts text using OCR from scanned PDFs."""
    text = ""
    try:
        images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
        for img in images:
            text += pytesseract.image_to_string(img, lang='hun')
    except Exception as e:
        print(f"Error performing OCR: {e}")
    return text


amounts = []
for file in os.listdir('./')[:]:
    if file[-3:] != 'pdf':
        continue
    
    file = 'S321.pdf'
    pdf_content = extract_text_from_pdf(file)
    
    if len(pdf_content) < 100:
        pdf_content = extract_text_via_ocr(file)        
    
#    gpt_prompt = "I send you an extract of a pdf bill invoice in hungarian. Your job is to find the final, full amount the invoice is about: " + pdf_content + ". Output the final total amount as an integer and nothing else! If you cant find the final total amount of the invoice, output 0."
    gpt_prompt = "I send you an extract of a pdf bill invoice in hungarian. Your job is to find the final several data from the invoice: " + pdf_content + ". Output the following in order: 1) the name of the partner, 2) the invoice number, 3) the total gross amount of the full invoice, 4) the total net amount of the invoice, 5) the total VAT (ÁFA in hungarian) of the invoice. Output these values (1 and 2 as strings, 3, 4 and 5 as integers) separated by ; and nothing else!"
    
    client = OpenAI(api_key=openai.api_key)
    
    response = client.chat.completions.create(
        model='gpt-4o', 
        messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": gpt_prompt}],
        max_tokens = 50,
        temperature=0,
        timeout=30)
    
    gpt_description = response.choices[0].message.content.strip()
    
    amounts.append([file, gpt_description])
    
    print(file)
    
