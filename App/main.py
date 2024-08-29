from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import functions  # Import the functions from functions.py

app = FastAPI(
    title="Drowsy Guard Detection API",
    description="Obtain drowsiness state value out of an image/video and return image and JSON result",
    version="0.0.1",
)

origins = ["http://localhost", "http://localhost:8080", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_bytes, detected_classes = functions.process_image(image_bytes)
    return JSONResponse(content={"detected_classes": detected_classes})

@app.post("/detect_and_image/")
async def detect_image_and_return(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_bytes, detected_classes = functions.process_image(image_bytes)
    return StreamingResponse(img_bytes, media_type="image/jpeg")

@app.get("/video_stream/")
async def video_stream():
    return StreamingResponse(functions.process_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detected_class_stream/")
async def detected_class_stream():
    return StreamingResponse(functions.process_detected_classes_stream(), media_type="text/event-stream")