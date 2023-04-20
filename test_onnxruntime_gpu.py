import  onnxruntime as ort
import torch

print(f"onnxruntime device: {ort.get_device()}") # output: GPU

print(f'ort avail providers: {ort.get_available_providers()}') # output: ['CUDAExecutionProvider', 'CPUExecutionProvider']

ort_session = ort.InferenceSession('./weights/antelopev2/glintr100.onnx', providers=["CUDAExecutionProvider"])

print(ort_session.get_providers()) # output: ['CPUExecutionProvider']

print(ort.get_device())

print(torch.backends.cudnn.version())