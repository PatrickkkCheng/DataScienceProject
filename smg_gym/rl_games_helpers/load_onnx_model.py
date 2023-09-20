import os
import numpy as np

import onnx
import onnxruntime as ort


saved_model_path = 'runs/smg_gaiting/dr_on'
onnx_path = os.path.join(
    saved_model_path, 'nn', 'smg_gaiting.onnx'
)

onnx_model = onnx.load(onnx_path)

# Check that the model is well formed
onnx.checker.check_model(onnx_model)

ort_model = ort.InferenceSession(onnx_path)

outputs = ort_model.run(
    None,
    {"obs": np.zeros((1, 96)).astype(np.float32)},
)
print(outputs)
