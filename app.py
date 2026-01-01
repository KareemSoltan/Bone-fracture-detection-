from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2


app = FastAPI(title="Bone Fracture Detection")


MODEL_PATH = "/content/fracture_bbox_model.keras"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_bbox(image: Image.Image, bbox):
    """
    bbox = [x1, y1, x2, y2] normalized between 0 and 1
    """
    img = np.array(image)
    h, w, _ = img.shape

    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)


    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return Image.fromarray(img)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bone Fracture Detection</title>
        <style>
            body { font-family: Arial; text-align: center; background: #f4f6f8; }
            .box { margin: auto; margin-top: 50px; padding: 20px; width: 300px;
                   background: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,.1); }
            button { padding: 10px; background: #007bff; color: white; border: none;
                     border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>Bone Fracture Detection</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br><br>
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    input_tensor = preprocess(image)
    bbox = model.predict(input_tensor)[0]


    output_image = draw_bbox(image, bbox)


    buffer = io.BytesIO()
    output_image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Result</title>
        <style>
            body {{ text-align: center; font-family: Arial; background: #f4f6f8; }}
            img {{ margin-top: 20px; max-width: 90%; border: 1px solid #ccc; }}
            a {{ display: inline-block; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h2>Prediction Result</h2>
        <img src="data:image/png;base64,{encoded_image}">
        <br>
        <a href="/">Upload another image</a>
    </body>
    </html>
    """
