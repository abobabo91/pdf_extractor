


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


MODEL_PRICES = {
    "gpt-4.1": {"input": 1.25, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-nano": {"input": 0.05, "output": 0.40},
}


def df_ready(df, required_cols=None):
    """Ellenőrzi, hogy a df nem üres és (opcionálisan) tartalmazza a kötelező oszlopokat."""
    if not isinstance(df, pd.DataFrame) or df is None or df.empty:
        return False
    if required_cols:
        return all(col in df.columns for col in required_cols)
    return True

def need_msg(missing_list):
    bullets = "\n".join([f"- {m}" for m in missing_list])
    st.warning(f"Az összefűzéshez a következők hiányoznak vagy üresek:\n{bullets}")

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
        for page in pdf_reader.pages[:]:
            pdf_content += page.extract_text() or ""
    except Exception as e:
        st.error(f"Hiba a(z) {file_name} fájl olvasásakor: {e}")
        return None

    if len(pdf_content.strip()) < 100:
        pdf_content = ""
        try:
            uploaded_file.seek(0)
            images = pdf2image.convert_from_bytes(uploaded_file.read())
#            images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path = r"C:\poppler-24.08.0\Library\bin") local
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
    """Generates a clear, structured GPT prompt for invoice data extraction."""
    return (
        "You are given the extracted text of a Hungarian invoice PDF. "
        "The PDF may contain multiple invoices merged together. "
        "Your task is to extract the following **9 data fields** for each invoice:\n\n"
        "1. Seller name (string)\n"
        "2. Buyer name (string)\n"
        "3. Invoice number (string)\n"
        "4. Invoice date (string, e.g. '2024.04.01')\n"
        "5. Total gross amount (integer)\n"
        "6. Total net amount (integer)\n"
        "7. VAT amount (integer)\n"
        "8. Currency (string: 'HUF' or 'EUR')\n"
        "9. Exchange rate (integer, use 1 if invoice is in HUF)\n\n"
        "**Important formatting instructions:**\n"
        "- Use semicolon (`;`) to separate the 9 fields.\n"
        "- Use **one line per invoice**.\n"
        "- Do **not** include field numbers (e.g. '1)', '2)' etc.) in the output.\n"
        "- Write all numeric fields as plain integers (e.g. `1500000`).\n"
        "- **Do not use thousands separators** (e.g. `.`) or decimal commas (`,`).\n"
        "- Note: In Hungarian, decimal separators are commas (`,`) instead of dots (`.`), "
        "but you must ignore this and always output plain integers.\n"
        "- Do **not** include any explanation, headings, or extra text — just the data rows.\n\n"
        "Extracted text:\n"
        f"{text}"
    )


def extract_data_with_gpt(file_name, text, model_name):
    """A kiválasztott GPT modellel kinyeri a struktúrált adatokat a PDF szövegből."""
    gpt_prompt = generate_gpt_prompt(text)

    try:
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": gpt_prompt}],
            max_completion_tokens=5000,
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

            cleaned_parts = [re.sub(r"^\s*\d+\)\s*", "", p.strip()) for p in parts]
            parsed_rows.append([file_name] + cleaned_parts)

        return parsed_rows, count_tokens(gpt_prompt)

    except Exception as e:
        st.error(f"A {model_name} feldolgozás sikertelen volt: {file_name} – {e}")
        return [], 0



def normalize_number(value):
    """Converts numeric-looking strings or floats to int. Removes all formatting."""
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return int(round(value))
        # Remove thousands separators ('.' or ',' or space), allow decimals
        cleaned = str(value).replace(" ", "").replace(",", "").replace(".", "")
        return int(cleaned)
    except:
        return None



def compare_with_tolerance(val1, val2, tolerance=500):
    try:
        val1 = normalize_number(val1)
        val2 = normalize_number(val2)
        if val1 is None or val2 is None:
            return False
        return abs(val1 - val2) <= tolerance
    except:
        return False

def get_minta_amount(row, huf_col="Érték", eur_col="Érték deviza", currency_col="Devizanem"):
    """Returns the value in correct currency column based on Devizanem."""
    try:
        dev = str(row[currency_col]).strip().upper()
        if dev == "EUR":
            return normalize_number(row[eur_col])
        else:
            return normalize_number(row[huf_col])
    except:
        return None


def compare_gpt_with_minta(df_minta, df_extracted, invoice_col_minta="Bizonylatszám", invoice_col_extracted="Számlaszám", tolerance=5):
    # Merge on invoice number
    df_merged = pd.merge(df_minta, df_extracted, how="outer", left_on=invoice_col_minta, right_on=invoice_col_extracted)

    # Compare amounts
    df_merged["Bruttó egyezik?"] = df_merged.apply(
        lambda row: compare_with_tolerance(
            get_minta_amount(row, huf_col="Érték", eur_col="Érték deviza", currency_col="Devizanem"),
            normalize_number(row["Bruttó ár"]),
            tolerance
        ),
        axis=1
    )

    # Optional: add summary column
    df_merged["Minden egyezik?"] = df_merged["Bruttó egyezik?"].apply(lambda x: "✅ Igen" if x else "❌ Nem")

    return df_merged


def merge_with_minta(df_extracted, df_minta, invoice_col_extracted="Számlaszám", invoice_col_minta="Bizonylatszám"):
    df_merged = pd.merge(df_minta, df_extracted, how='outer', left_on=invoice_col_minta, right_on=invoice_col_extracted)
    matched = df_merged[invoice_col_extracted].notna().sum()
    total = len(df_minta)
    unmatched = total - matched
    match_rate = round(100 * matched / total, 2)
    
    stats = {
        "Összes minta sor": total,
        "Találatok száma": matched,
        "Hiányzó találatok": unmatched,
        "Egyezési arány (%)": match_rate
    }

    return df_merged, stats



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
    
    selected_model = st.selectbox(
        "Válassz modellt az adatkinyeréshez:",
        ["gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
        index=0,
        help="Árak per 1M token:\n"
             "- gpt-4.1: \$1.25 input / \$10 output\n"
             "- gpt-4o: \$2.50 input / \$10 output\n"
             "- gpt-4.1-mini: \$0.25 input / \$2 output\n"
             "- gpt-4.1-nano: \$0.05 input / \$0.40 output"
    )


    
    # 1) PDF feldolgozás
    if st.button("📑 Adatkinyerés a PDF-ből"):  
        st.session_state.extracted_text_from_invoice = []      
        if uploaded_files:
            if len(uploaded_files) > 200:
                st.write("⚠️ Az első 100 fájl kerül feldolgozásra.")
    
            for uploaded_file in uploaded_files[:200]:
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
                extracted_rows, tokens = extract_data_with_gpt(file_name, pdf_content, selected_model)
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
    price_input = st.session_state.number_of_tokens * MODEL_PRICES[selected_model]["input"] / 1_000_000
    st.write(f"💰 A becsült feldolgozási költség eddig: **${price_input:.2f}** ({selected_model})")



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
 
    
 
st.title("📄 Ellenőrzések")

col_left, col_right = st.columns([1, 1])  # nagyobb bal oldali hasáb

with col_left:
    st.subheader("📎 Kinyert adatok összefűzése és ellenőrzése: Mintavétel")
    
    invoice_colname_minta = "Bizonylatszám"
        
    if st.button("🔗 Összefűzés és ellenőrzés a Mintavétel excellel"):
        # Előfeltételek
        missing = []
        if not df_ready(st.session_state.df_extracted, required_cols=["Számlaszám", "Nettó ár"]):
            missing.append("Kinyert adatok (PDF feldolgozás)")
        if not df_ready(st.session_state.df_minta, required_cols=["Bizonylatszám", "Érték", "Érték deviza", "Devizanem"]):
            missing.append("Mintavétel Excel")
    
        if missing:
            need_msg(missing)
        else:
            try:
                df_minta = st.session_state.df_minta.copy()
                # Fejlécek biztonságos tisztítása (nem használ .str-t)
                df_minta.columns = [str(c).strip() for c in df_minta.columns]
                df_minta["Bizonylatszám"] = df_minta["Bizonylatszám"].astype(str)
    
                df_gpt = st.session_state.df_extracted.copy()
                df_gpt["Számlaszám"] = df_gpt["Számlaszám"].astype(str)
    
                # ⬅️ GPT balra, Minta jobbra
                df_merged_minta = pd.merge(
                    df_gpt,
                    df_minta,
                    how="left",
                    left_on="Számlaszám",
                    right_on="Bizonylatszám"
                )
    
                # Nettó összehasonlítás a mintával
                df_merged_minta["Nettó egyezik?"] = df_merged_minta.apply(
                    lambda row: compare_with_tolerance(
                        get_minta_amount(row, huf_col="Érték", eur_col="Érték deviza", currency_col="Devizanem"),
                        normalize_number(row["Nettó ár"]),
                        tolerance=5
                    ),
                    axis=1
                )
    
                df_merged_minta["Minden egyezik?"] = df_merged_minta["Nettó egyezik?"].apply(
                    lambda x: "✅ Igen" if x else "❌ Nem"
                )
    
                st.session_state.df_merged_minta = df_merged_minta
    
                # 📊 Statisztika
                total = len(df_merged_minta)
                matched = (df_merged_minta["Minden egyezik?"] == "✅ Igen").sum()
                match_rate = round(100 * matched / total, 2)
    
                st.session_state.stats_minta = {
                    "Összes számla": total,
                    "Minden egyezés": matched,
                    "Egyezési arány (%)": match_rate
                }
    
                st.success("✅ Összefűzés és ellenőrzés a Mintavétellel kész!")
    
            except Exception as e:
                st.error(f"Váratlan hiba történt a Mintavétel összefűzés során: {e}")

    
    if "df_merged_minta" in st.session_state:
        st.write("📄 **Összefűzött és ellenőrzött táblázat – Mintavétel:**")
        st.dataframe(st.session_state.df_merged_minta)
    
        csv_minta = st.session_state.df_merged_minta.to_csv(index=False).encode("utf-8")
    
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_merged_minta.to_excel(writer, sheet_name='Minta', index=False)
            writer.close()
            st.download_button(
                label="📥 Letöltés Excel (Mintavétel)",
                data=buffer,
                file_name='merged_minta.xlsx',
                mime='application/vnd.ms-excel'
            )
    
        st.markdown("### 📊 Statisztika – Mintavétel ellenőrzés")
        for k, v in st.session_state.stats_minta.items():
            st.write(f"**{k}:** {v}")
    


with col_right:
    st.subheader("📎 Kinyert adatok összefűzése és ellenőrzése: NAV")
    
    if st.button("🔗 Összefűzés és ellenőrzés a NAV excellel"):
        # Előfeltételek
        missing = []
        if not df_ready(st.session_state.df_extracted, required_cols=["Számlaszám", "Bruttó ár", "Nettó ár", "ÁFA"]):
            missing.append("Kinyert adatok (PDF feldolgozás)")
        # A NAV fájlban legalább a számlaszám és az összegek valamelyike legyen
        nav_required = ["számlasorszám"]  # az összegek neve változhat, ezért ezt lent rugalmasan kezeljük
        if not df_ready(st.session_state.df_nav, required_cols=nav_required):
            missing.append("NAV Excel")
    
        if missing:
            need_msg(missing)
        else:
            try:
                df_nav = st.session_state.df_nav.copy()
                df_nav.columns = [str(c).strip() for c in df_nav.columns]
                df_nav["számlasorszám"] = df_nav["számlasorszám"].astype(str)
    
                df_gpt = st.session_state.df_extracted.copy()
                df_gpt["Számlaszám"] = df_gpt["Számlaszám"].astype(str)
    
                df_merged_nav = pd.merge(
                    df_gpt,
                    df_nav,
                    how="left",
                    left_on="Számlaszám",
                    right_on="számlasorszám"
                )
    
                # Oszlopnevek rugalmas keresése (különböző exportok)
                brutto_col = "bruttó érték" if "bruttó érték" in df_merged_nav.columns else ("bruttó érték Ft" if "bruttó érték Ft" in df_merged_nav.columns else None)
                netto_col  = "nettóérték"  if "nettóérték"  in df_merged_nav.columns else ("nettóérték Ft"  if "nettóérték Ft"  in df_merged_nav.columns else None)
                afa_col    = "adóérték"    if "adóérték"    in df_merged_nav.columns else ("adóérték Ft"    if "adóérték Ft"    in df_merged_nav.columns else None)
    
                # Ha nincs egyik összegoszlop sem, adjunk barátságos jelzést
                amount_missing = []
                if brutto_col is None:
                    amount_missing.append("bruttó érték (NAV)")
                if netto_col is None:
                    amount_missing.append("nettóérték (NAV)")
                if afa_col is None:
                    amount_missing.append("adóérték (NAV)")
                if amount_missing:
                    need_msg(amount_missing)
    
                # Összegellenőrzések (csak ha van megfelelő NAV oszlop)
                df_merged_nav["Bruttó egyezik?"] = df_merged_nav.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(brutto_col)) if brutto_col else None,
                        normalize_number(row.get("Bruttó ár")),
                    ),
                    axis=1
                )
    
                df_merged_nav["Nettó egyezik?"] = df_merged_nav.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(netto_col)) if netto_col else None,
                        normalize_number(row.get("Nettó ár")),
                    ),
                    axis=1
                )
    
                df_merged_nav["ÁFA egyezik?"] = df_merged_nav.apply(
                    lambda row: compare_with_tolerance(
                        normalize_number(row.get(afa_col)) if afa_col else None,
                        normalize_number(row.get("ÁFA")),
                    ),
                    axis=1
                )
    
                df_merged_nav["Minden egyezik?"] = df_merged_nav.apply(
                    lambda row: "✅ Igen" if (row["Bruttó egyezik?"] and row["Nettó egyezik?"] and row["ÁFA egyezik?"]) else "❌ Nem",
                    axis=1
                )
    
                st.session_state.df_merged_nav = df_merged_nav
    
                # Statisztika
                total = len(df_merged_nav)
                matched_all = (df_merged_nav["Minden egyezik?"] == "✅ Igen").sum()
                match_rate = round(100 * matched_all / total, 2)
    
                st.session_state.stats_nav = {
                    "Összes számla": total,
                    "Minden egyezés": matched_all,
                    "Teljes egyezési arány (%)": match_rate
                }
    
                st.success("✅ NAV fájllal való összefűzés és ellenőrzés kész!")
    
            except Exception as e:
                st.error(f"Váratlan hiba történt a NAV összefűzés során: {e}")

    
    if "df_merged_nav" in st.session_state:
        st.write("📄 **Összefűzött és ellenőrzött táblázat – NAV:**")
        st.dataframe(st.session_state.df_merged_nav)
    
        # Excel letöltés előkészítés
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_merged_nav.to_excel(writer, sheet_name='NAV összehasonlítás', index=False)
            writer.close()
    
        st.download_button(
            label="📥 Letöltés Excel (NAV)",
            data=buffer,
            file_name="merged_nav.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
        st.markdown("### 📊 Statisztika – NAV összehasonlítás")
        for k, v in st.session_state.stats_nav.items():
            st.write(f"**{k}:** {v}")
    



asdd = """
st.subheader("📎 Kinyert adatok összefűzése: Karton EZT EGYELŐRE NEM CSINÁLTAM MEG")

if st.button("🔗 Összefűzés a Kartonnal"):
    try:
        df_merged_karton = pd.merge(
            st.session_state.df_extracted,
            st.session_state.df_karton,
            how="left",
            left_on="Számlaszám",
            right_on=invoice_colname_karton
        )

        matched_karton = df_merged_karton[invoice_colname_karton].notna().sum()
        total_karton = len(st.session_state.df_extracted)
        unmatched_karton = total_karton - matched_karton
        match_rate_karton = round(100 * matched_karton / total_karton, 2)

        st.session_state.df_merged_karton = df_merged_karton
        st.session_state.stats_karton = {
            "Összes számla": total_karton,
            "Karton egyezés": matched_karton,
            "Hiányzó egyezés": unmatched_karton,
            "Egyezési arány (%)": match_rate_karton
        }

        st.success("✅ Karton összefűzés kész!")

    except Exception as e:
        st.error(f"❌ Hiba történt a Karton összefűzés során: {e}")

if "df_merged_karton" in st.session_state:
    st.write("📄 **Összefűzött táblázat – Karton:**")
    st.dataframe(st.session_state.df_merged_karton)

    csv_karton = st.session_state.df_merged_karton.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Letöltés CSV (Karton)", csv_karton, "merged_karton.csv", "text/csv")

    st.markdown("### 📊 Statisztika – Karton összefűzés")
    for k, v in st.session_state.stats_karton.items():
        st.write(f"**{k}:** {v}")
"""








local_test = r"""

os.listdir('./')


import tomllib

secrets_path = r"C:\Users\abele\.streamlit\secrets.toml"

with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)

openai.api_key = secrets["OPENAI_API_KEY"]

MODEL = "gpt-4.1"  # cheapest useful model
PDF_FILE = "9601013656.pdf"  # <-- replace with your own file path


with open(PDF_FILE, "rb") as f:
    text = extract_text_from_pdf(f)


# 2) GPT extraction
rows, tokens_used = extract_data_with_gpt(PDF_FILE, text, MODEL)


# 3) Create DataFrame
df = pd.DataFrame(rows, columns=[
    "Fájl", "Eladó", "Vevő", "Számlaszám", "Számla dátum",
    "Bruttó ár", "Nettó ár", "ÁFA", "Pénznem", "Árfolyam"
])





"""



local_test = r"""
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
