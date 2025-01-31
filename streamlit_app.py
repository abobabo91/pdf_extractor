import streamlit as st
import pandas as pd
import numpy as np
from st_files_connection import FilesConnection

import openai
from openai import OpenAI


openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]



st.title('Pdf extractor')

st.write('Hi!')
