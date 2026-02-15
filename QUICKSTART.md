# Mobile VLA API Server - Quick Start Guide

**Branch**: `inference-integration`  
**Quantization**: BitsAndBytes INT8  
**Target**: Jetson Orin / RTX GPU

---

## 🚀 Quick Start (5분 설치)

### 1. Clone Repository
```bash
git clone git@github.com-vla:minuum/vla.git
cd vla
git checkout inference-integration
```

### 2. Install Dependencies
```bash
# Python 3.10+ required
pip install -r requirements-inference.txt
```

### 3. Download Model (Already included)
```bash
# Chunk5 Best model이 이미 포함되어 있음
ls runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/
# epoch_epoch=06-val_loss=val_loss=0.067.ckpt 확인
```

### 4. Set API Key
```bash
# Option 1: Export
export VLA_API_KEY="your-secret-api-key-here"

# Option 2: Use secrets.sh
echo 'export VLA_API_KEY="your-secret-api-key-here"' > secrets.sh
source secrets.sh
```

### 5. Start Server
```bash
# Foreground (for testing)
python3 -m uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000

# Background (for production)
nohup python3 -m uvicorn Mobile_VLA.inference_server:app \
  --host 0.0.0.0 --port 8000 > logs/api_server.log 2>&1 &
```

### 6. Test
```bash
# Health check
curl http://localhost:8000/health

# Quick test
python3 scripts/test_api_inference_complete.py
```

---

## 📋 System Requirements

### Minimum
- **OS**: Ubuntu 20.04+ / Jetson Linux
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA 11.8+
- **Memory**: 16GB RAM
- **GPU Memory**: 4GB+ (INT8 uses ~1.8GB)

### Recommended
- **GPU**: RTX 3060 12GB+ or Jetson Orin 16GB
- **CUDA**: 12.0+
- **Memory**: 32GB RAM

### Jetson Specific
- **Jetson Orin 16GB**: ✅ Perfect match
- **Jetson Xavier 16GB**: ✅ Compatible
- **Jetson Nano**: ❌ Insufficient memory

---

## 🔧 Configuration

### Default Settings
```python
# inference_server.py
checkpoint_path = "runs/.../epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
```

### Custom Model (Optional)
```bash
# Environment variable로 변경 가능
export VLA_CHECKPOINT_PATH="/path/to/your/model.ckpt"
export VLA_CONFIG_PATH="/path/to/your/config.json"
```

---

## 📊 Performance

### Billy Server (RTX A5000)
- **GPU Memory**: 1.80 GB
- **Inference**: 495 ms/call
- **Rate**: 2.0 Hz
- **18 consecutive**: 9.6 seconds

### Expected on Jetson Orin
- **GPU Memory**: ~1.8 GB (same)
- **Inference**: 500-600 ms/call
- **Rate**: 1.7-2.0 Hz
- **18 consecutive**: 10-11 seconds

---

## 🌐 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "gpu_memory": {
        "allocated_gb": 1.80,
        "device_name": "NVIDIA RTX A5000"
    }
}
```

### Inference
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "instruction": "Move forward to the target"
  }'
```

Response:
```json
{
    "action": [0.5, 0.0],
    "latency_ms": 495.6,
    "model_name": "mobile_vla_chunk5_20251217",
    "strategy": "receding_horizon",
    "source": "inferred",
    "buffer_status": {}
}
```

---

## 🐍 Python Client Example

```python
import requests
import base64
from PIL import Image
import io

# API 설정
API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

# 이미지 로드
image = Image.open("robot_view.jpg")
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()

# Inference 요청
response = requests.post(
    f"{API_URL}/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "image": img_b64,
        "instruction": "Move forward"
    }
)

result = response.json()
action = result["action"]  # [linear_x, linear_y]
print(f"Action: {action}")
```

---

## 🔍 Troubleshooting

### 1. ModuleNotFoundError: bitsandbytes
```bash
pip install bitsandbytes==0.43.1
```

### 2. CUDA out of memory
```bash
# GPU memory 확인
nvidia-smi

# 다른 프로세스 종료
pkill -f python
```

### 3. Import error: action_buffer
```bash
# 이미 수정됨 (from Mobile_VLA.action_buffer import ActionBuffer)
# 최신 코드 pull 필요
git pull origin inference-integration
```

### 4. Checkpoint not found
```bash
# 경로 확인
ls runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/

# 환경 변수로 지정
export VLA_CHECKPOINT_PATH="/path/to/your/checkpoint.ckpt"
```

---

## 🎯 Production Deployment

### 1. Systemd Service (Optional)
```bash
# /etc/systemd/system/vla-api.service
[Unit]
Description=Mobile VLA API Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/vla
Environment="VLA_API_KEY=your-secret-key"
ExecStart=/usr/bin/python3 -m uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vla-api
sudo systemctl start vla-api
```

### 2. Docker (Optional)
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY requirements-inference.txt .
RUN pip install -r requirements-inference.txt

COPY . .
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "Mobile_VLA.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📚 Documentation

- **API Spec**: `docs/API_SPECIFICATION_INT8.md`
- **Architecture**: `docs/BITSANDBYTES_ARCHITECTURE_20251224.md`
- **Performance**: `docs/ROBOT_DRIVING_18STEPS_TEST_20251224.md`
- **Comparison**: `docs/QUANTIZATION_FINAL_COMPARISON_20251224.md`

---

## 🔐 Security

### API Key Management
```bash
# Generate secure key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in environment
export VLA_API_KEY="generated-key"

# Or use secrets.sh (gitignored)
echo 'export VLA_API_KEY="generated-key"' > secrets.sh
source secrets.sh
```

### Network Security
- **Recommended**: Use Tailscale VPN
- **Alternative**: Firewall rules (allow only specific IPs)
- **Not recommended**: Expose to public internet

---

## 🎉 Verification

### Quick Test
```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Single inference
python3 scripts/test_api_inference_complete.py

# 3. 18 consecutive (robot simulation)
python3 scripts/test_robot_driving_18steps.py
```

### Expected Results
- ✅ Health: `"status": "healthy"`
- ✅ GPU Memory: ~1.8 GB
- ✅ Latency: ~500 ms
- ✅ 18 consecutive: ~9-10 seconds

---

## 📞 Support

### Logs
```bash
# Server logs
tail -f logs/api_server.log

# Test logs
ls logs/*.log
```

### Common Issues
1. **GPU not detected**: Check CUDA installation
2. **Slow inference**: Check GPU utilization (nvidia-smi)
3. **Memory leak**: Monitor GPU memory over time

---

**Last Updated**: 2025-12-24  
**Branch**: inference-integration  
**Status**: Production Ready ✅
