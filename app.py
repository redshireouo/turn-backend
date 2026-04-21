from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import subprocess
import imageio_ffmpeg

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
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "model" / "weights.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")

model = None


def get_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    return model


def convert_video_to_web_mp4(input_path: str, output_path: str):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    subprocess.run(command, check=True)


@app.get("/")
def root():
    return {"message": "backend is running"}


@app.head("/")
def root_head():
    return


@app.post("/predict")
async def predict(request: Request, video: UploadFile = File(...)):
    yolo_model = get_model()

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}_{video.filename}"

    # 1. 儲存上傳影片
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 2. 建立這次推論輸出資料夾
    run_output_dir = RESULT_DIR / file_id
    run_output_dir.mkdir(exist_ok=True)

    # 3. 跑模型推論
    results = yolo_model.predict(
        source=str(save_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=True,
        project=str(RESULT_DIR),
        name=file_id,
        exist_ok=True,
        verbose=False,
        stream=True
    )

    has_sign = False
    has_square = False
    detections = []

    # 4. 讀取每幀結果
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

    # 5. 播報邏輯
    announcement = ""
    if has_sign:
        announcement = "前方有待轉牌，要待轉"
    elif has_square:
        announcement = "前方有待轉格，可能要待轉"

    # 6. 找模型輸出的影片
    output_video_url = None
    mp4_file = None
    avi_file = None

    for file in run_output_dir.iterdir():
        suffix = file.suffix.lower()
        if suffix == ".mp4":
            mp4_file = file
        elif suffix == ".avi":
            avi_file = file

    # 7. 優先使用 mp4；沒有就把 avi 轉成瀏覽器可播的 mp4
    if mp4_file is not None:
        output_video_url = str(request.base_url).rstrip("/") + f"/results/{file_id}/{mp4_file.name}"

    elif avi_file is not None:
        converted_mp4 = run_output_dir / "output_web.mp4"
        convert_video_to_web_mp4(str(avi_file), str(converted_mp4))
        output_video_url = str(request.base_url).rstrip("/") + f"/results/{file_id}/{converted_mp4.name}"

    print("run_output_dir:", run_output_dir)
    print("files in run_output_dir:", [f.name for f in run_output_dir.iterdir()])
    print("output_video_url:", output_video_url)

    return {
        "has_sign": has_sign,
        "has_square": has_square,
        "announcement": announcement,
        "output_video_url": output_video_url,
        "detections": detections[:50]
    }