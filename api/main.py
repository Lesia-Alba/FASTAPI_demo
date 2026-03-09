import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from utils.model_func import (
    load_text_model,
    load_image_model,
    predict_text,
    predict_image,
)

logger = logging.getLogger("uvicorn.info")

text_model = None
image_model = None


class TextInput(BaseModel):
    text: str


class TextResponse(BaseModel):
    label: str
    prob: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_model
    global image_model

    text_model = load_text_model()
    logger.info("Text model loaded")

    image_model = load_image_model()
    logger.info("Image model loaded")

    yield

    del text_model, image_model
    logger.info("Models unloaded")


app = FastAPI(
    title="BERT + YOLO11 API",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/clf_text", response_model=TextResponse)
def predict_text_endpoint(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    result = predict_text(data.text, text_model)

    return TextResponse(
        label=result["label"],
        prob=result["prob"]
    )


@app.post("/clf_image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    image_bytes = await file.read()
    result = predict_image(image_bytes, image_model)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)