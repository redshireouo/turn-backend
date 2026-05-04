from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import io
import shutil
import uuid
import traceback

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
        print("Loading YOLO model...")
        model = YOLO(str(MODEL_PATH))
        print("YOLO model loaded.")
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

    try:
        yolo_model = get_model()

        file_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{file_id}_{video.filename}"

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print(f"saved video: {save_path}")

        results = yolo_model.predict(
            source=str(save_path),
            conf=0.25,
            iou=0.45,
            imgsz=416,
            save=False,
            verbose=False,
            stream=True,
            device="cpu",
            max_det=10,
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

    except Exception as e:
        print("predict failed")
        print("error:", repr(e))
        traceback.print_exc()
        raise e


@app.post("/predict-frame")
async def predict_frame(image: UploadFile = File(...)):
    print("=== /predict-frame called ===")

    try:
        yolo_model = get_model()

        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        pil_image.thumbnail((512, 512))
        analysis_width, analysis_height = pil_image.size
        print("image size after thumbnail:", pil_image.size)

        results = yolo_model.predict(
            source=pil_image,
            conf=0.25,
            iou=0.45,
            imgsz=320,
            save=False,
            verbose=False,
            device="cpu",
            max_det=5,
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

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if class_name == "sign":
                    has_sign = True
                elif class_name == "square":
                    has_square = True

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    }
                })

        announcement = ""
        if has_sign:
            announcement = "前方有待轉牌，要待轉"
        elif has_square:
            announcement = "前方有待轉格，可能要待轉"

        return {
            "has_sign": has_sign,
            "has_square": has_square,
            "announcement": announcement,
            "image_size": {
                "width": analysis_width,
                "height": analysis_height
            },
            "detections": detections
        }

    except Exception as e:
        print("predict-frame failed")
        print("error:", repr(e))
        traceback.print_exc()
        raise e