import base64
import uvicorn
import numpy as np
import cv2
import insightface
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Pydantic model for request body
class EmbeddingRequest(BaseModel):
    # Detection size parameters with defaults; Internally for every image insightfaces re-sizes the image to these values
    img_width: int = 640
    img_height: int = 640
    
    # Base64-encoded image
    img_b64: str

app = FastAPI(title="Face Embedding FastAPI Interface")

origins = ['null']         
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Face Embedding FastAPI Interface!"}

@app.post("/faceEmbeddingB64")
def get_face_embedding(req: EmbeddingRequest):
    """
    POST JSON data:
    {
        "img_width": 640,
        "img_height": 640,
        "img_b64": "<base64-encoded-image>"
    }

    Returns the first face's embedding (list of floats).
    """
    # 1. Decode the base64 image
    try:
        image_data = base64.b64decode(req.img_b64)
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))    
    """
    # For debugging: save the *raw decoded bytes* to a file
    debug_raw_path = "debug_raw_decoded.jpg"
    with open(debug_raw_path, "wb") as f:
        f.write(image_data)
    """
    try:
        # 2. Convert bytes to a NumPy array, then decode via OpenCV
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Could not decode the image. Check file format or base64 string."}

        # 3. Initialize (or Re-configure) FaceAnalysis with user-specified detection size.
        #    This is done here for demonstration, but may be slow in practice if repeated often.
        face_analyzer = insightface.app.FaceAnalysis()

        # For GPU use, set ctx_id=0, else set ctx_id=-1 for CPU
        face_analyzer.prepare(ctx_id=-1, det_size=(req.img_width, req.img_height))
        
        # 4. Detect faces
        faces = face_analyzer.get(img)
        if not faces:
            return {"error": "No face detected in the image."}
        
        # Take the first face's normalized embedding
        # Taking only the first embeddings means that we are assuming only one face is passed to the API
        embedding = faces[0].normed_embedding
        embedding_list = embedding.tolist()  # Convert to Python list

        return {"embedding": embedding_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/faceEmbeddingImg")
async def get_face_embedding_img(
    img_file: UploadFile = File(...),
    img_width: int = Form(640),
    img_height: int = Form(640),
):
    """
    Receives a file (multipart) + form fields for width and height.
    Example of a multipart form data request with:
        - image file in "file"
        - integer form fields "width" and "height"
    """
    try:
        # 1. Read file contents
        image_bytes = await img_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Invalid image or file format."}

        # 2. Initialize face analyzer with custom detection size
        face_analyzer = insightface.app.FaceAnalysis()
        face_analyzer.prepare(ctx_id=-1, det_size=(img_width, img_height))

        # 3. Detect faces and get embedding
        faces = face_analyzer.get(img)
        if not faces:
            return {"error": "No face detected."}

        embedding = faces[0].normed_embedding.tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
"""