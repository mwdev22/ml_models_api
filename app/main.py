import io
import pickle

import numpy as np
import PIL.ImageOps
import PIL.Image
from sklearn.ensemble import RandomForestClassifier

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .extensions import ML_MODELS_DIR



# reading bytes
# single model for now, to be modified
with open(f'{ML_MODELS_DIR}handwritten_digits.pkl', 'rb') as f:
    model: RandomForestClassifier = pickle.load(f)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.post('/predict-digits')
async def home(file: UploadFile = File(...)):
    contents = await file.read()
    # open byte str of contents, conver it to grayscale
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L') 
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize(28, 28, PIL.Image.ANTIALIAS)
    img_array = np.array(pil_image).reshape(1,-1)
    prediction = model.predict(img_array)
    return {'prediction':int(prediction[0])}