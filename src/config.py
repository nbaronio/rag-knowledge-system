"""
Shared configuration and clients used across the pipeline.
"""

import os
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gemini-3.6-flash"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])