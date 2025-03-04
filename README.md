# Face Embedding FastAPI Interface

This repository provides a FastAPI-based interface for extracting face embeddings from images using InsightFace. The API supports both base64-encoded images and file uploads.

## Features
- Extract face embeddings from images.
- Supports base64-encoded images and direct file uploads.
- Configurable face detection size.
- Runs on CPU by default, but can be configured for GPU acceleration.

## Prerequisites
Ensure you have the following dependencies installed:

- Python 3.8+
- FastAPI
- Uvicorn
- OpenCV (`cv2`)
- NumPy
- InsightFace
- Pydantic

## Running the Application

To start the FastAPI application, run:

Local running for tests:
```sh
uvicorn main_server:app --host 0.0.0.0 --port 8000
```

Background running for effective server deployment:
```sh
nohup uvicorn main_server:app --host 0.0.0.0 --port 8000 &
```

## API Endpoints

### Root Endpoint
- `GET /` - Returns a welcome message.

### Face Embedding Extraction
- `POST /faceEmbeddingB64` - Extracts a face embedding from a base64-encoded image.
- `POST /faceEmbeddingImg` - Extracts a face embedding from an uploaded image file.

## Example Usage

### Extract Embedding from Base64 Image
```sh
curl -X POST "http://localhost:8000/faceEmbeddingB64" -H "Content-Type: application/json" -d '{
  "img_width": 640,
  "img_height": 640,
  "img_b64": "<base64-encoded-image>"
}'
```

### Extract Embedding from Uploaded File
```sh
curl -X POST "http://localhost:8000/faceEmbeddingImg" \
  -F "img_file=@path/to/image.jpg" \
  -F "img_width=640" \
  -F "img_height=640"
```

## Configuration

Modify the following parameters to optimize performance:

- `ctx_id=-1`: Runs on CPU.
- `ctx_id=0`: Enables GPU acceleration (requires a compatible CUDA environment).
- `img_width` and `img_height`: Resize values for face detection.

## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss the changes.
