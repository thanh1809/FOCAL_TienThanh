import os
import sys
import glob
import json
import torch
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add web_app and src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from inference import engine, ACTIVITY_CLASSES

app = FastAPI(title="FOCAL Multimodal Sensing Web UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "PAMAP2_1of10", "samples")

PRESET_SAMPLES = [
    {"file": "subject103_Protocol_000730.pt", "label_id": 3, "name": "walking", "vi": "Đi bộ", "icon": "🚶", "tag": "Sample 01"},
    {"file": "subject103_Protocol_000000.pt", "label_id": 0, "name": "lying", "vi": "Nằm nghỉ ngơi", "icon": "🛌", "tag": "Sample 02"},
    {"file": "subject103_Protocol_000111.pt", "label_id": 1, "name": "sitting", "vi": "Ngồi ghế", "icon": "🪑", "tag": "Sample 03"},
    {"file": "subject103_Protocol_000255.pt", "label_id": 2, "name": "standing", "vi": "Đứng yên", "icon": "🧍", "tag": "Sample 04"},
    {"file": "subject103_Protocol_000630.pt", "label_id": 10, "name": "ascending_stairs", "vi": "Leo cầu thang", "icon": "🪜", "tag": "Sample 05"},
    {"file": "subject103_Protocol_000600.pt", "label_id": 11, "name": "descending_stairs", "vi": "Xuống cầu thang", "icon": "📉", "tag": "Sample 06"},
    {"file": "subject103_Protocol_000497.pt", "label_id": 12, "name": "vacuum_cleaning", "vi": "Hút bụi nhà", "icon": "🧹", "tag": "Sample 07"},
    {"file": "subject103_Protocol_000357.pt", "label_id": 13, "name": "ironing", "vi": "Là quần áo", "icon": "👔", "tag": "Sample 08"}
]

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return "<h1>FOCAL Web UI</h1><p>Vui lòng đợi khởi tạo static files...</p>"

@app.get("/api/samples")
async def get_samples():
    """List available presets and available samples"""
    existing_presets = []
    for s in PRESET_SAMPLES:
        full_p = os.path.join(SAMPLES_DIR, s["file"])
        if os.path.exists(full_p):
            existing_presets.append(s)
    
    # Also find if user uploaded any sample files
    user_samples_dir = os.path.join(PROJECT_ROOT, "weights", "uploads")
    user_files = []
    if os.path.exists(user_samples_dir):
        for f in os.listdir(user_samples_dir):
            if f.endswith((".pt", ".csv", ".json")):
                user_files.append({"file": f, "is_uploaded": True})

    return {
        "presets": existing_presets,
        "user_uploads": user_files,
        "total_pamap2_samples": len(glob.glob(os.path.join(SAMPLES_DIR, "*.pt"))),
        "classes": ACTIVITY_CLASSES
    }

@app.get("/api/weights")
async def get_weights_status():
    """Return model status and available weights in weights/ folder"""
    weights = engine.find_available_weights()
    return {
        "is_real_model": engine.is_real_model,
        "active_weight_file": engine.active_weight_file,
        "model_type": engine.model_type,
        "available_weights": weights,
        "device": str(engine.device),
        "weights_dir": os.path.abspath(engine.weights_dir)
    }

@app.post("/api/weights/reload")
async def reload_weights(weight_filename: Optional[str] = Form(None), model_name: Optional[str] = Form("SW_Transformer")):
    """Trigger weight reload when user downloads best_model.pt from Colab"""
    success, msg = engine.reload_model(model_name=model_name, weight_filename=weight_filename)
    return {
        "success": success,
        "message": msg,
        "is_real_model": engine.is_real_model,
        "active_weight_file": engine.active_weight_file
    }

def format_sensor_waveform(data_dict):
    """
    Extract waveforms for Chart.js:
    Wrist Acc (X, Y, Z), Gyro (X, Y, Z), Mag (X, Y, Z)
    Shape of each tensor: [3, 9, 40] -> Flatten intervals to 9*40 = 360 points (2 seconds)
    """
    wrist = data_dict.get("wrist", {})
    waveforms = {}
    
    for mod in ["acc", "gyro", "mag"]:
        if mod in wrist:
            tensor = wrist[mod]
            if isinstance(tensor, torch.Tensor):
                arr = tensor.cpu().numpy()
            else:
                arr = np.array(tensor)
            
            # If shape is [3, 9, 40], flatten [9, 40] -> 360 points
            if arr.ndim == 3:
                c, i, s = arr.shape
                flattened = arr.reshape(c, i * s)
                # Sample down slightly if needed, e.g. every 2 points or full 360 points
                waveforms[mod] = {
                    "x": [round(float(v), 3) for v in flattened[0].tolist()],
                    "y": [round(float(v), 3) for v in flattened[1].tolist()],
                    "z": [round(float(v), 3) for v in flattened[2].tolist()],
                    "total_samples": i * s
                }
    return waveforms

@app.post("/api/predict/sample/{filename}")
async def predict_preset_sample(filename: str):
    """Load a sample .pt and predict"""
    # Check in PAMAP2 samples or uploads
    file_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(file_path):
        upload_path = os.path.join(PROJECT_ROOT, "weights", "uploads", filename)
        if os.path.exists(upload_path):
            file_path = upload_path
        else:
            raise HTTPException(status_code=404, detail="File sample không tồn tại")

    sample = torch.load(file_path, map_location="cpu")
    data_dict = sample.get("data", {})
    true_label = sample.get("label", None)
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.item() if true_label.numel() == 1 else int(true_label[0])

    # Run inference
    inference_result = engine.predict(data_dict, true_label=true_label)
    
    # Extract waveform visualization
    waveforms = format_sensor_waveform(data_dict)

    # Truth info
    true_info = None
    if true_label is not None and 0 <= int(true_label) < len(ACTIVITY_CLASSES):
        true_info = ACTIVITY_CLASSES[int(true_label)]

    return {
        "filename": filename,
        "true_label": true_info,
        "prediction": inference_result["prediction"],
        "pipeline_steps": inference_result["pipeline_steps"],
        "model_status": inference_result["model_status"],
        "waveforms": waveforms
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle upload of .pt sample or model weight"""
    filename = file.filename
    content = await file.read()
    
    # If file is a model weight (best_model.pt)
    if filename.endswith(".pt") and ("model" in filename.lower() or "weight" in filename.lower() or "best" in filename.lower()):
        save_path = os.path.join(PROJECT_ROOT, "weights", filename)
        with open(save_path, "wb") as f:
            f.write(content)
        # Reload engine
        engine.reload_model(weight_filename=filename)
        return {
            "type": "model_weight",
            "message": f"Đã lưu trọng số model: {filename}",
            "active_weight_file": engine.active_weight_file,
            "is_real_model": engine.is_real_model
        }
    
    # Otherwise treat as data sample
    uploads_dir = os.path.join(PROJECT_ROOT, "weights", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    save_path = os.path.join(uploads_dir, filename)
    with open(save_path, "wb") as f:
        f.write(content)
        
    try:
        sample = torch.load(save_path, map_location="cpu")
        data_dict = sample.get("data", {})
        true_label = sample.get("label", None)
        inference_result = engine.predict(data_dict, true_label=true_label)
        waveforms = format_sensor_waveform(data_dict)
        return {
            "type": "data_sample",
            "filename": filename,
            "prediction": inference_result["prediction"],
            "pipeline_steps": inference_result["pipeline_steps"],
            "model_status": inference_result["model_status"],
            "waveforms": waveforms
        }
    except Exception as e:
        return {
            "type": "uploaded_raw",
            "filename": filename,
            "message": f"Đã lưu file thành công: {filename}. (Lưu ý: để dự đoán ngay, file cần đúng định dạng sample PyTorch .pt của PAMAP2)"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Khởi động FOCAL Web UI tại: http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
