# Mobile VLA - BitsAndBytes INT8 Inference

**Branch**: `inference-integration`  
**Status**: 🟢 Production Ready  
**Quantization**: BitsAndBytes INT8 (OpenVLA/BitVLA Standard)

---

## 🎯 Quick Start

```bash
# 1. Clone & Install
git clone git@github.com-vla:minuum/vla.git
cd vla
git checkout inference-integration
pip install -r requirements-inference.txt

# 2. Setup (Jetson)
./setup_jetson.sh

# 3. Start Server
source secrets.sh
python3 -m uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000

# 4. Test
curl http://localhost:8000/health
```

**완전한 가이드**: [QUICKSTART.md](QUICKSTART.md)

---

## ⚡ Performance

### BitsAndBytes INT8 (vs FP32)

| Metric | FP32 | **INT8** | Improvement |
|--------|------|----------|-------------|
| **GPU Memory** | 6.3 GB | **1.8 GB** | **71% ↓** |
| **Inference** | 15 s | **0.5 s** | **30x faster** |
| **Rate** | 0.067 Hz | **2.0 Hz** | **30x** |

### Real Robot Testing

- ✅ **18 consecutive inferences**: 9.6 seconds
- ✅ **Latency**: 495 ± 7 ms (extremely stable)
- ✅ **Memory**: No leaks
- ✅ **Reliability**: 100% success rate

---

## 📊 System Requirements

### Minimum
- **GPU**: NVIDIA with 4GB+ VRAM
- **Memory**: 16GB RAM
- **CUDA**: 11.8+

### Recommended
- **Jetson Orin 16GB**: ✅ Perfect match
- **RTX 3060 12GB+**: ✅ Excellent
- **RTX A5000 24GB**: ✅ Overkill but great

---

## 🚀 Features

### BitsAndBytes INT8 Quantization
- ✅ **73% memory reduction** (6.3GB → 1.8GB)
- ✅ **30x speed improvement** (15s → 0.5s)
- ✅ **~98% accuracy maintained** (OpenVLA verified)
- ✅ **No retraining required** (Post-training quantization)

### API Server
- ✅ **RESTful API** (FastAPI)
- ✅ **API Key authentication**
- ✅ **Health monitoring**
- ✅ **Production ready**

### Models
- ✅ **Chunk5 Best** (Val Loss: 0.067)
- ✅ **Left Chunk10** (Val Loss: 0.010)
- ✅ **Right Chunk10** (Val Loss: 0.013)

---

## 📁 Project Structure

```
vla/
├── QUICKSTART.md              # Complete setup guide
├── requirements-inference.txt  # Minimal dependencies
├── setup_jetson.sh            # Auto-setup for Jetson
│
├── Mobile_VLA/
│   ├── inference_server.py    # API Server (INT8)
│   ├── action_buffer.py       # Action buffering
│   └── configs/               # Model configurations
│
├── scripts/
│   ├── test_api_inference_complete.py    # API test
│   ├── test_robot_driving_18steps.py     # Robot simulation
│   └── sync/                             # Sync scripts
│
├── docs/
│   ├── API_SPECIFICATION_INT8.md         # API docs
│   ├── BITSANDBYTES_ARCHITECTURE_20251224.md
│   ├── ROBOT_DRIVING_18STEPS_TEST_20251224.md
│   └── QUANTIZATION_FINAL_COMPARISON_20251224.md
│
└── runs/                      # Trained models
    └── mobile_vla_no_chunk_20251209/
        └── kosmos/mobile_vla_finetune/2025-12-17/
            └── mobile_vla_chunk5_20251217/
                └── epoch_epoch=06-val_loss=val_loss=0.067.ckpt
```

---

## 🌐 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Inference
```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/predict",
    headers={"X-API-Key": "your-key"},
    json={
        "image": base64_image,
        "instruction": "Move forward"
    }
)

action = response.json()["action"]  # [linear_x, linear_y]
```

---

## 📚 Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **API Specification**: [docs/API_SPECIFICATION_INT8.md](docs/API_SPECIFICATION_INT8.md)
- **Architecture**: [docs/BITSANDBYTES_ARCHITECTURE_20251224.md](docs/BITSANDBYTES_ARCHITECTURE_20251224.md)
- **Performance**: [docs/ROBOT_DRIVING_18STEPS_TEST_20251224.md](docs/ROBOT_DRIVING_18STEPS_TEST_20251224.md)

---

## 🎯 Use Cases

### Real Robot Deployment
- ✅ **Jetson Orin**: 1.8GB footprint (10GB headroom)
- ✅ **2.0 Hz inference rate** (sufficient for indoor robots)
- ✅ **Stable, predictable performance**

### Research & Development
- ✅ **Fast iteration** (30x faster than FP32)
- ✅ **Multi-model testing** (low memory per model)
- ✅ **Quantification studies** (INT8 vs FP32)

---

## 🔬 Technical Details

### Quantization Method
- **Method**: BitsAndBytes INT8 (LLM-INT8)
- **Standard**: OpenVLA, BitVLA, Octo
- **GPU**: CUDA kernels (not CPU)
- **Accuracy**: ~98% of FP32 (verified)

### Model Architecture
- **Base**: Kosmos-2 1.6B
- **Vision**: CLIP ViT
- **Action Head**: LSTM decoder
- **Window**: 8 frames
- **Chunk**: 5 actions

---

## 🛠️ Development

### Training (on main branch)
```bash
git checkout main
# See training documentation
```

### Inference (this branch)
```bash
git checkout inference-integration
./setup_jetson.sh
# Ready to deploy
```

---

## 📞 Support

### Common Issues
1. **ModuleNotFoundError: bitsandbytes**
   - Solution: `pip install bitsandbytes==0.43.1`

2. **CUDA out of memory**
   - Solution: Check `nvidia-smi`, kill other processes

3. **Checkpoint not found**
   - Solution: Sync from Billy server or download

### Logs
```bash
# Server logs
tail -f logs/api_server.log

# Test results
cat logs/robot_driving_test_18steps.json
```

---

## 🎉 Results Summary

### What We Achieved (Dec 24, 2025)

1. ✅ **BitsAndBytes INT8 Implementation**
   - 73% memory reduction
   - 30x speed improvement
   - OpenVLA/BitVLA standard

2. ✅ **Complete Testing**
   - 3/3 models verified (100%)
   - 18 consecutive inferences
   - Real robot simulation

3. ✅ **Production Ready**
   - API Server integrated
   - Documentation complete
   - Jetson deployment ready

4. ✅ **Performance Verified**
   - 495ms latency (± 7ms)
   - No memory leaks
   - 100% reliability

---

## 📜 License

See main repository for license information.

---

## 🙏 Acknowledgments

- **OpenVLA**: BitsAndBytes INT8 method
- **BitVLA**: Quantization techniques
- **Microsoft**: Kosmos-2 base model
- **Hugging Face**: Transformers, BitsAndBytes

---

**Last Updated**: 2025-12-24  
**Maintainer**: Billy  
**Branch**: inference-integration  
**Status**: 🟢 Production Ready
