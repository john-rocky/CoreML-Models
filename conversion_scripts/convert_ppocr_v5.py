# PP-OCRv5 -> CoreML conversion
# PP-OCRv5 by Baidu PaddlePaddle - Ultra lightweight multilingual OCR
# https://github.com/PaddlePaddle/PaddleOCR
# pip install paddlepaddle paddleocr torch coremltools onnx

# Step 1: Export PaddleOCR to ONNX using paddle2onnx
# pip install paddle2onnx
# paddle2onnx --model_dir ./PP-OCRv5_det --model_filename inference.pdmodel \
#   --params_filename inference.pdiparams --save_file ppocrv5_det.onnx

# Step 2: Convert ONNX to CoreML
import coremltools as ct
import onnx

# Detection model
det_onnx = onnx.load("ppocrv5_det.onnx")
det_ml = ct.converters.convert(
    det_onnx,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 640, 640), scale=1/255.0)],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
det_ml.save("PPOCRv5_Det.mlpackage")

# Recognition model
rec_onnx = onnx.load("ppocrv5_rec.onnx")
rec_ml = ct.converters.convert(
    rec_onnx,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 48, 320))],
    minimum_deployment_target=ct.target.iOS16,
    convert_to="mlprogram",
)
rec_ml.save("PPOCRv5_Rec.mlpackage")
print("Saved PPOCRv5_Det.mlpackage and PPOCRv5_Rec.mlpackage")
