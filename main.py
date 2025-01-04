import os
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import shutil
import uuid

from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="outputs"), name="static")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 이미지 저장 디렉토리
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_to_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch


def interpolate_images(image1, image2, alpha):
    return cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)


def process_images(image_paths, num_interpolations=10):
    sketches = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {path}")
            continue
        sketch = convert_to_sketch(img)
        sketches.append(sketch)

    if len(sketches) < 2:
        return []

    output_paths = []
    for i in range(len(sketches) - 1):
        sketch1 = sketches[i]
        sketch2 = sketches[i + 1]

        sketch2 = cv2.resize(sketch2, (sketch1.shape[1], sketch1.shape[0]))

        alpha_values = [j / (num_interpolations + 1) for j in range(1, num_interpolations + 1)]
        for j, alpha in enumerate(alpha_values):
            interpolated = interpolate_images(sketch1, sketch2, alpha)
            output_filename = f'interpolated_sketch_{i + 1}_{i + 2}_{j + 1}.jpg'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, interpolated)
            output_paths.append(output_path)

    return output_paths


@app.post("/interpolate/")
async def create_interpolations(files: List[UploadFile] = File(...)):
    if len(files) < 2:
        return JSONResponse(content={"error": "At least 2 images are required"}, status_code=400)

    saved_files = []
    try:
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)

        interpolated_paths = process_images(saved_files)

        # 상대 URL 생성
        base_url = "/static/"  # 실제 서버 설정에 맞게 조정 필요
        relative_urls = [f"{base_url}{os.path.basename(path)}" for path in interpolated_paths]

        return JSONResponse(content={"interpolated_images": relative_urls})

    finally:
        # 업로드된 원본 파일 삭제
        for file_path in saved_files:
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

