

https://docs.mistral.ai/capabilities/OCR/basic_ocr/

WORKING EXAMPLE OF MISTRAL OCR PDF

import base64
import os
from mistralai import Mistral

def encode_pdf(pdf_path):
    """Encode the PDF to base64 string."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

pdf_path = "S321.pdf"  # ← Change this to your actual file
base64_pdf = encode_pdf(pdf_path)

api_key = "uhuY5m9xE2jjBryZZ46liqLakXf8t02p"  
client = Mistral(api_key=api_key)

response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}"
    },
    include_image_base64=False,
    document_annotation_format={"type": "json", "json_schema": {}}
)

print("✅ OCR Response:")
print(response.model_dump_json(indent=2))  




import base64
import os
from mistralai import Mistral

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None

# Path to your image
image_path = "S321.png"

# Getting the base64 string
base64_image = encode_image(image_path)


ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{base64_image}" 
    },
    include_image_base64=True
)

print("✅ OCR Response:")
print(ocr_response.model_dump_json(indent=2))  








-----------pdf extractorok


import pdfplumber

with pdfplumber.open("invoice.pdf") as pdf:
    page = pdf.pages[0]
    words = page.extract_words()  # Each word has x0, y0, x1, y1
    print(words)
    table = page.extract_table()
    print(table)



from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

for page_layout in extract_pages("invoice.pdf"):
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            print(element.get_text(), element.bbox)  # Text + (x0, y0, x1, y1)



import fitz  # PyMuPDF

doc = fitz.open("S126.pdf")
page = doc[0]
for block in page.get_text("dict")["blocks"]:
    print(block)







-----------image extractorok
import easyocr
reader = easyocr.Reader(['hu'])  # Add languages you need
result = reader.readtext('S321.png')







from paddleocr import PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang='hu')
result = ocr.predict('S321.png')

texts = result[0]['rec_texts']
scores = result[0]['rec_scores']

for text, score in zip(texts, scores):
    print(f"{text} ({score:.2f})")
    
    
full_text = '\n'.join(texts)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
