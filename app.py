from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import io
import shutil
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "weights.pt"

UPLOAD_DIR.mkdir(exist_ok=True)

model = None


def get_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    return model


@app.get("/")
def root():
    return {"message": "backend is running"}


@app.head("/")
def root_head():
    return


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    print("=== /predict called ===")

    yolo_model = get_model()

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}_{video.filename}"

    # 儲存上傳影片
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    print(f"saved video: {save_path}")

    # 影片推論：先不要輸出結果影片，先求雲端穩定
    results = yolo_model.predict(
        source=str(save_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=False,
        verbose=False,
        stream=True
    )

    has_sign = False
    has_square = False
    detections = []

    for frame_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = yolo_model.names[cls_id]

            if class_name == "sign":
                has_sign = True
            elif class_name == "square":
                has_square = True

            detections.append({
                "frame": frame_idx,
                "class": class_name,
                "confidence": round(conf, 3)
            })

    announcement = ""
    if has_sign:
        announcement = "前方有待轉牌，要待轉"
    elif has_square:
        announcement = "前方有待轉格，可能要待轉"

    print("predict finished successfully")
    print("has_sign:", has_sign, "has_square:", has_square, "detections:", len(detections))

    return {
        "has_sign": has_sign,
        "has_square": has_square,
        "announcement": announcement,
        "output_video_url": None,
        "detections": detections[:50]
    }


@app.post("/predict-frame")
async def predict_frame(image: UploadFile = File(...)):
    print("=== /predict-frame called ===")

    yolo_model = get_model()

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = yolo_model.predict(
        source=pil_image,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=False,
        verbose=False
    )

    has_sign = False
    has_square = False
    detections = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = yolo_model.names[cls_id]

            if class_name == "sign":
                has_sign = True
            elif class_name == "square":
                has_square = True

            detections.append({
                "class": class_name,
                "confidence": round(conf, 3)
            })

    announcement = ""
    if has_sign:
        announcement = "前方有待轉牌，要待轉"
    elif has_square:
        announcement = "前方有待轉格，可能要待轉"

    print("predict-frame finished successfully")
    print("has_sign:", has_sign, "has_square:", has_square, "detections:", len(detections))

    return {
        "has_sign": has_sign,
        "has_square": has_square,
        "announcement": announcement,
        "detections": detections
    }