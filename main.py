from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from services.yolo_service import process_video_with_yolo
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "YOLOv5 Video Processing API"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video to the static folder
        video_path = f"static/{file.filename}"
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Process the video using YOLOv5
        output_video_path = process_video_with_yolo(video_path)

        # Return the processed video
        return FileResponse(output_video_path, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
