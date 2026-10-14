import os
import random
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa

app = FastAPI(
    title="AcousticSpace API",
    description="Deepfake Audio Detection via Room Impulse Response (RIR) & Audio Spectrogram Transformer (AST)",
    version="1.0.0"
)

# Enable CORS for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
