#!/usr/bin/env bash

#pip install --no-cache-dir "tf_keras" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0" "onnx>=1.12.0" "onnx2tf>=1.26.3" "onnxslim>=0.1.31" "tflite_support" "onnxruntime" --extra-index-url https://pypi.ngc.nvidia.com
#pip install --no-cache-dir "tf_keras" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0" "onnx>=1.12.0" "onnx2tf" "onnxslim>=0.1.31" "tflite_support" "onnxruntime" --extra-index-url https://pypi.ngc.nvidia.com
yolo export model=yolov8n.pt format=tflite int8=True