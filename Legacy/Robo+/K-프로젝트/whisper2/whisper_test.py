#whisper_tset.py

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "TheoJo/whisper-tiny-ko"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# /app/input.wav 등을 mount해서 사용
waveform, sr = torchaudio.load("/app/input.wav")
waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
predicted_ids = model.generate(inputs)
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"인식 결과: {text}")