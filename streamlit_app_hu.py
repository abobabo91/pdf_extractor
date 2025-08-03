import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import os
import PyPDF2
import pdf2image
import pytesseract
from io import BytesIO
import openai
from openai import OpenAI
import tiktoken
import re

def count_tokens(text, model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)

def replace_successive_duplicates(df, column_to_compare, columns_to_delete):
    result = df.copy()
    col = column_to_compare
    mask = result[col] == result[col].shift()
    for col in columns_to_delete:
        result.loc[mask, col] = np.nan
    return result

def extract_text_from_pdf(uploaded_file):
    """Megpróbálja szövegesen kinyerni a PDF tartalmát, OCR-rel kiegészítve, ha szükséges."""
    file_name = uploaded_file.name
    pdf_content = ""

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_content += page.extract_text() or ""
    except Exception as e:
        st.error(f"Hiba a(z) {file_name} fájl olvasásakor: {e}")
        return None

    if len(pdf_content.strip()) < 100:
        pdf_content = ""
        try:
            uploaded_file.seek(0)
            images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path = r"C:\poppler-24.08.0\Library\bin")
            for img in images:
                pdf_content += pytesseract.image_to_string(img, lang="hun")
        except Exception as e:
            st.error(f"OCR hiba a(z) {file_name} fájlnál: {e}")
            return None

    if len(pdf_content) > 500000:
        st.write(file_name + " túl hosszú, csak az első 5000 karakter kerül feldolgozásra.")
        pdf_content = pdf_content[:5000]

    return pdf_content


def generate_gpt_prompt(text):
    """Összeállítja a promptot a GPT számára."""
    return ("""I send you an extract of a pdf bill invoice in Hungarian. It may contain several invoices merged into one pdf. Your job is to find several data from the invoice/invoices: """ 
            + text +  """. Output the following in order: 
            1) the name of the seller, 
            2) the name of the buyer, 
            3) the invoice number, 
            4) the date of the invoice,
            5) the total gross amount of the full invoice, 
            6) the total net amount of the invoice, 
            7) the total VAT (ÁFA in Hungarian) of the invoice,
            8) the currency used on the invoice (Ft or Eur),
            9) the HUF/EUR currency exchange rate (if the invoice is in Ft, then write 1).
            Be careful that in Hungarian the decimal separator is ',' instead of '.', and the thousands separator is '.', instead of ','.
            Output these 9 values (1, 2, 3, 4 and 8 as strings, 5, 6, 7 and 9 as integers) separated by ; and each invoice in new line as many invoices there are and nothing else!""")



def extract_data_with_gpt(file_name, text):
    """GPT-4o segítségével kinyeri a struktúrált adatokat a PDF szövegből – akár több számlára."""
    gpt_prompt = generate_gpt_prompt(text)

    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model='gpt-4o', 
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": gpt_prompt}],
            max_tokens=10000,
            temperature=0,
            timeout=30
        )

        raw_output = response.choices[0].message.content.strip()
        rows = raw_output.split("\n")

        parsed_rows = []
        for row in rows:
            parts = row.strip().split(";")
            if len(parts) != 9:
                st.warning(f"Hibás sor ({file_name}): {row}")
                continue

            # 🧼 Tisztítsuk meg az egyes mezőket a felesleges prefixektől pl. "1) "
            cleaned_parts = [re.sub(r"^\s*\d+\)\s*", "", p.strip()) for p in parts]
            
            parsed_rows.append([file_name] + parts)

        return parsed_rows, count_tokens(gpt_prompt)

    except Exception as e:
        st.error(f"A GPT-4 feldolgozás sikertelen volt: {file_name} – {e}")
        return [], 0



# Inicializáljuk a session state változókat
if 'extracted_text_from_invoice' not in st.session_state:
    st.session_state.extracted_text_from_invoice = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'df_extracted' not in st.session_state:
    st.session_state.df_extracted = pd.DataFrame()
if 'df_minta' not in st.session_state:
    st.session_state.df_minta = pd.DataFrame()
if 'df_nav' not in st.session_state:
    st.session_state.df_nav = pd.DataFrame()
if 'df_karton' not in st.session_state:
    st.session_state.df_karton = pd.DataFrame()
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = pd.DataFrame()
if 'df_merged_full' not in st.session_state:
    st.session_state.df_merged_full = pd.DataFrame()
if 'number_of_tokens' not in st.session_state:
    st.session_state.number_of_tokens = 0

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]

st.title("📄 Számlaadat-kinyerő alkalmazás")

col_pdf, col_excel = st.columns([1, 1])  # nagyobb bal oldali hasáb

with col_pdf:
    st.subheader("📂 PDF-ek kinyerése")
    
    st.write("1) Tölts fel egy vagy több **magyar nyelvű számlát (PDF)**, amelyekből a rendszer kiolvassa a legfontosabb adatokat.")
    
    # 0) Fájl feltöltő
    uploaded_files = st.file_uploader("📤 PDF fájlok feltöltése", type=["pdf"], accept_multiple_files=True)
    
    # 1) PDF feldolgozás
    if st.button("📑 Szövegkinyerés a PDF-ből"):  
        st.session_state.extracted_text_from_invoice = []      
        if uploaded_files:
            if len(uploaded_files) > 100:
                st.write("⚠️ Az első 100 fájl kerül feldolgozásra.")
    
            for uploaded_file in uploaded_files[:100]:
                file_name = uploaded_file.name
                pdf_text = extract_text_from_pdf(uploaded_file)
    
                if pdf_text is None:
                    continue
    
                st.session_state.extracted_text_from_invoice.append([file_name, pdf_text])
        else:
            st.warning("⚠️ Kérlek, tölts fel legalább egy PDF fájlt.")
    
        # 2) GPT adatkinyerés
        st.session_state.extracted_data = []
        if st.session_state.extracted_text_from_invoice:
            for file_name, pdf_content in st.session_state.extracted_text_from_invoice:
                extracted_rows, tokens = extract_data_with_gpt(file_name, pdf_content)
                if extracted_rows:
                    st.session_state.extracted_data.extend(extracted_rows)
                    st.session_state.number_of_tokens += tokens
                    st.write(f"{file_name} feldolgozása kész.")

    
        if st.session_state.extracted_data:
            st.session_state.df_extracted = pd.DataFrame(
                st.session_state.extracted_data,
                columns=["Fájlnév", "Eladó", "Vevő", "Számlaszám", "Számla kelte", "Bruttó ár", "Nettó ár", "ÁFA", "Deviza", "Árfolyam"]
            )
            st.session_state.df_extracted["Számlaszám"] = st.session_state.df_extracted["Számlaszám"].astype(str)
            st.session_state.df_extracted["1"] = np.nan

    
    if len(st.session_state.df_extracted) > 0:        
        st.write("✅ **Adatok kinyerve!** Az alábbi táblázat tartalmazza az eredményeket:")
        st.dataframe(st.session_state.df_extracted)
    
        extract_csv = st.session_state.df_extracted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Kinyert adatok letöltése CSV-ben", extract_csv, "kinyert_adatok.csv", "text/csv", key="letoltes-csv")
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_extracted.to_excel(writer, sheet_name='Adatok', index=False)
            writer.close()
            st.download_button(
                label="📥 Kinyert adatok letöltése Excelben",
                data=buffer,
                file_name='kinyert_adatok.xlsx',
                mime='application/vnd.ms-excel'
            )


    # Token ár becslés
    price = st.session_state.number_of_tokens * 2.5 / 1_000_000
    st.write(f"💰 A becsült feldolgozási költség eddig: **${price:.4f}** (GPT-4o)")


with col_excel:
    
    st.subheader("📂 Excel fájlok betöltése")
    
    
    # --- Excel fájlok betöltése ---
    st.markdown("1) Töltsd fel a **Mintavétel** Excel fájlt:")
    uploaded_excel_file_minta = st.file_uploader(
        "📤 Mintavétel Excel feltöltése",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon a 10. sortól induljanak, és legyen 'Bizonylatszám' nevű oszlop."
    )
    
    
    if uploaded_excel_file_minta:
        try:
            st.session_state.df_minta = pd.read_excel(uploaded_excel_file_minta, skiprows=range(1, 9))
            st.session_state.df_minta.columns = list(st.session_state.df_minta.iloc[0])
            st.session_state.df_minta = st.session_state.df_minta.iloc[1:]
            st.session_state.df_minta["Bizonylatszám"] = st.session_state.df_minta["Bizonylatszám"].astype(str)
        except:
            st.warning("❌ Nem sikerült beolvasni a Mintavétel fájlt.")
    
    if len(st.session_state.df_minta) > 0:        
        st.write("✅ **Mintavétel betöltve!** Első néhány sor:")
        st.dataframe(st.session_state.df_minta.head(5))
    
    
    
    # NAV fájl
    st.markdown("2) Töltsd fel a **NAV** Excel fájlt:")
    uploaded_excel_file_nav = st.file_uploader(
        "📤 NAV Excel feltöltése",
        type=["xlsx"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon a 6. sortól induljanak, és legyen 'számlasorszám' nevű oszlop."
    )    
    if uploaded_excel_file_nav:
        try:
            st.session_state.df_nav = pd.read_excel(uploaded_excel_file_nav, skiprows=5)
            st.session_state.df_nav["számlasorszám"] = st.session_state.df_nav["számlasorszám"].astype(str)
        except:
            st.warning("❌ Nem sikerült beolvasni a NAV fájlt.")
    
    if len(st.session_state.df_nav) > 0:        
        st.write("✅ **NAV fájl betöltve!** Első néhány sor:")
        st.dataframe(st.session_state.df_nav.head(5))
    
    
    
    # Karton fájl
    st.markdown("3) Töltsd fel a **Karton** Excel fájlt:")
    uploaded_excel_file_karton = st.file_uploader(
        "📤 Karton Excel feltöltése",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        help="Az adatok az első munkalapon az A1 cellától induljanak."
    )    
    
    default_invoice_column_karton = "Bizonylat"
    custom_colname_enabled_karton = st.checkbox("🔧 Saját oszlopnév megadása a számlaszámhoz a Karton excelben (Alapértelmezett: 'Bizonylat')", value=False)
    
    if custom_colname_enabled_karton:
        invoice_colname_karton = st.text_input("Add meg a számlaszámot tartalmazó oszlop nevét a Karton excelben:", value=default_invoice_column_karton)
    else:
        invoice_colname_karton = default_invoice_column_karton
    
    if uploaded_excel_file_karton:
        try:
            st.session_state.df_karton = pd.read_excel(uploaded_excel_file_karton)
            st.session_state.df_karton[invoice_colname_karton] = st.session_state.df_karton[invoice_colname_karton].astype(str)
        except:
            st.warning("❌ Nem sikerült beolvasni a Karton fájlt.")
    
    if len(st.session_state.df_karton) > 0:        
        st.write("✅ **Karton betöltve!** Első néhány sor:")
        st.dataframe(st.session_state.df_karton.head(5))
 
    
st.subheader("📂 A kinyert adatok és az Excel fájlok összefűzése")

# Összefűzés
if len(st.session_state.df_extracted) > 0 and len(st.session_state.df_minta) > 0 and len(st.session_state.df_karton) > 0:
    st.write("4) Kinyert adatok és Excel fájlok összefűzése:")
    if st.button("🔗 Összefűzés"):  
        try:
            df_temp = pd.merge(st.session_state.df_minta, st.session_state.df_extracted, how='outer', left_on='Bizonylatszám', right_on='Számlaszám')
            st.session_state.df_merged = df_temp 
            nr_of_columns = len(df_temp.columns)
            df_temp = pd.merge(df_temp, st.session_state.df_karton, how='left', left_on='Bizonylatszám', right_on=invoice_colname_karton)
            st.session_state.df_merged_full = replace_successive_duplicates(df_temp, 'Bizonylatszám', df_temp.columns[:nr_of_columns])
        except:
            st.warning("❌ Hiba történt az összefűzés során.")

if len(st.session_state.df_merged) > 0:
    st.write("✅ **Összefűzés kész!**")
    st.dataframe(st.session_state.df_merged)

    csv = st.session_state.df_merged.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Összefűzött adatok letöltése (CSV)", csv, "osszeadott_adatok.csv", "text/csv", key="download-merged-csv")
    
    csv_full = st.session_state.df_merged_full.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Teljes összefűzött adatok (CSV)", csv_full, "osszeadott_teljes.csv", "text/csv", key="download-merged-full-csv")
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        st.session_state.df_merged.to_excel(writer, sheet_name='Munka1', index=False)
        st.session_state.df_merged_full.to_excel(writer, sheet_name='Munka2', index=False)
        writer.close()
        st.download_button(
            label="📥 Letöltés Excel formátumban",
            data=buffer,
            file_name='osszeadott_adatok.xlsx',
            mime='application/vnd.ms-excel'
        )





local_test = """
df_extracted = pd.read_csv('extract_data.csv')
df_minta = pd.read_excel('Mintavétel_költségek_Sonneveld és ellenïrzés.xlsx', sheet_name='Mintavétel', skiprows = range(1, 9))
df_minta.columns = list(df_minta.iloc[0])
df_minta = df_minta.iloc[1:]
df_minta["Bizonylatszám"] = df_minta["Bizonylatszám"].astype(str)
df_karton = pd.read_excel('Könyvelési karton 2024_Sonneveld Kft.xlsx', sheet_name='Munka1')

df_temp = pd.merge(df_minta, df_extracted, how='outer', left_on='Bizonylatszám', right_on='Számlaszám')
nr_of_columns = len(df_temp.columns)

df_temp = pd.merge(df_temp, df_karton, how='left', left_on='Bizonylatszám', right_on='Bizonylat')

column_to_compare = 'Bizonylatszám'
columns_to_delete = df_temp.columns[:nr_of_columns]
df_merged = replace_successive_duplicates(df_temp, column_to_compare, columns_to_delete)

"""
