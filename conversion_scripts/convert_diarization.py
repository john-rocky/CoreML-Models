"""
Convert pyannote/segmentation-3.0 to CoreML.

Architecture from ONNX graph analysis:
  SincConv(80, 251, stride=10) → Abs → MaxPool(3) → InstanceNorm → LeakyRelu(0)
  Conv(60, 5) → MaxPool(3) → InstanceNorm → LeakyRelu(0)
  Conv(60, 5) → MaxPool(3) → InstanceNorm → LeakyRelu(0)
  Transpose → LSTM(128, uni) → LSTM(128, uni) concat → reverse
  Linear(256,128) → LeakyRelu → Linear(128,128) → LeakyRelu → Linear(128,7) → LogSoftmax
"""

import torch
import torch.nn as nn
import numpy as np
import os
import onnx
import onnxruntime as ort
import coremltools as ct


class PyanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav_norm = nn.InstanceNorm1d(1, affine=True, eps=1e-5)
        self.sinc_conv = nn.Conv1d(1, 80, 251, stride=10, padding=0, bias=False)
        self.norm0 = nn.InstanceNorm1d(80, affine=True, eps=1e-5)
        self.conv1 = nn.Conv1d(80, 60, 5, padding=0)
        self.norm1 = nn.InstanceNorm1d(60, affine=True, eps=1e-5)
        self.conv2 = nn.Conv1d(60, 60, 5, padding=0)
        self.norm2 = nn.InstanceNorm1d(60, affine=True, eps=1e-5)

        # 4-layer BiLSTM
        self.lstm = nn.LSTM(60, 128, num_layers=4, batch_first=False, bidirectional=True)

        self.linear0 = nn.Linear(256, 128)
        self.linear1 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, 7)

    def forward(self, x):
        x = self.wav_norm(x)

        x = self.sinc_conv(x)
        x = torch.abs(x)
        x = nn.functional.max_pool1d(x, 3, stride=3)
        x = self.norm0(x)
        x = nn.functional.leaky_relu(x, 0.01)

        x = self.conv1(x)
        x = nn.functional.max_pool1d(x, 3, stride=3)
        x = self.norm1(x)
        x = nn.functional.leaky_relu(x, 0.01)

        x = self.conv2(x)
        x = nn.functional.max_pool1d(x, 3, stride=3)
        x = self.norm2(x)
        x = nn.functional.leaky_relu(x, 0.01)

        x = x.permute(2, 0, 1)  # [B, C, T] → [T, B, C]
        x, _ = self.lstm(x)

        x = nn.functional.leaky_relu(self.linear0(x), 0.01)
        x = nn.functional.leaky_relu(self.linear1(x), 0.01)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=-1)

        x = x.permute(1, 0, 2)  # [T, B, 7] → [B, T, 7]
        return x


def reorder_lstm_gates(w):
    """ONNX gate order [i,o,f,c] → PyTorch [i,f,g,o]"""
    chunks = torch.chunk(torch.from_numpy(w), 4, dim=0)
    return torch.cat([chunks[0], chunks[2], chunks[3], chunks[1]], dim=0)


def load_weights(model, onnx_path):
    m = onnx.load(onnx_path)
    W = {w.name: onnx.numpy_helper.to_array(w) for w in m.graph.initializer}

    with torch.no_grad():
        model.wav_norm.weight.copy_(torch.from_numpy(W['sincnet.wav_norm1d.weight']))
        model.wav_norm.bias.copy_(torch.from_numpy(W['sincnet.wav_norm1d.bias']))
        model.sinc_conv.weight.copy_(torch.from_numpy(W['/sincnet/conv1d.0/Concat_2_output_0']))
        model.norm0.weight.copy_(torch.from_numpy(W['sincnet.norm1d.0.weight']))
        model.norm0.bias.copy_(torch.from_numpy(W['sincnet.norm1d.0.bias']))
        model.conv1.weight.copy_(torch.from_numpy(W['sincnet.conv1d.1.weight']))
        model.conv1.bias.copy_(torch.from_numpy(W['sincnet.conv1d.1.bias']))
        model.norm1.weight.copy_(torch.from_numpy(W['sincnet.norm1d.1.weight']))
        model.norm1.bias.copy_(torch.from_numpy(W['sincnet.norm1d.1.bias']))
        model.conv2.weight.copy_(torch.from_numpy(W['sincnet.conv1d.2.weight']))
        model.conv2.bias.copy_(torch.from_numpy(W['sincnet.conv1d.2.bias']))
        model.norm2.weight.copy_(torch.from_numpy(W['sincnet.norm1d.2.weight']))
        model.norm2.bias.copy_(torch.from_numpy(W['sincnet.norm1d.2.bias']))

        # 4-layer BiLSTM
        lstm_weights = [
            ('onnx::LSTM_784', 'onnx::LSTM_785', 'onnx::LSTM_783'),  # layer 0
            ('onnx::LSTM_827', 'onnx::LSTM_828', 'onnx::LSTM_826'),  # layer 1
            ('onnx::LSTM_870', 'onnx::LSTM_871', 'onnx::LSTM_869'),  # layer 2
            ('onnx::LSTM_913', 'onnx::LSTM_914', 'onnx::LSTM_912'),  # layer 3
        ]
        for layer_idx, (w_key, r_key, b_key) in enumerate(lstm_weights):
            for d in range(2):
                sfx = '' if d == 0 else '_reverse'
                getattr(model.lstm, f'weight_ih_l{layer_idx}{sfx}').copy_(
                    reorder_lstm_gates(W[w_key][d]))
                getattr(model.lstm, f'weight_hh_l{layer_idx}{sfx}').copy_(
                    reorder_lstm_gates(W[r_key][d]))
                b = W[b_key][d]
                getattr(model.lstm, f'bias_ih_l{layer_idx}{sfx}').copy_(
                    reorder_lstm_gates(b[:512].reshape(4, 128)).reshape(-1))
                getattr(model.lstm, f'bias_hh_l{layer_idx}{sfx}').copy_(
                    reorder_lstm_gates(b[512:].reshape(4, 128)).reshape(-1))

        # Linear (ONNX MatMul is [in, out], PyTorch Linear is [out, in])
        model.linear0.weight.copy_(torch.from_numpy(W['onnx::MatMul_915'].T))
        model.linear0.bias.copy_(torch.from_numpy(W['linear.0.bias']))
        model.linear1.weight.copy_(torch.from_numpy(W['onnx::MatMul_916'].T))
        model.linear1.bias.copy_(torch.from_numpy(W['linear.1.bias']))
        model.classifier.weight.copy_(torch.from_numpy(W['onnx::MatMul_917'].T))
        model.classifier.bias.copy_(torch.from_numpy(W['classifier.bias']))


def main():
    onnx_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--onnx-community--pyannote-segmentation-3.0/snapshots/")
    snapshot = os.listdir(onnx_dir)[0]
    onnx_path = os.path.join(onnx_dir, snapshot, "onnx", "model.onnx")

    print("Building PyanNet...")
    model = PyanNet()
    load_weights(model, onnx_path)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Verify
    dummy = np.random.randn(1, 1, 160000).astype(np.float32)
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {'input_values': dummy})[0]
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy)).numpy()

    print(f"ONNX:    shape={onnx_out.shape}, range=[{onnx_out.min():.2f}, {onnx_out.max():.2f}]")
    print(f"PyTorch: shape={pt_out.shape}, range=[{pt_out.min():.2f}, {pt_out.max():.2f}]")

    if onnx_out.shape != pt_out.shape:
        print(f"Shape mismatch!")
        return

    diff = np.abs(onnx_out - pt_out).max()
    print(f"Max diff: {diff:.6f} {'OK' if diff < 0.05 else 'MISMATCH'}")

    if diff > 0.05:
        return

    # Trace & convert
    print("\nTracing...")
    traced = torch.jit.trace(model, torch.from_numpy(dummy))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, 1, 160000))],
        outputs=[ct.TensorType(name="speaker_logits")],
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "CoreML-Models"
    mlmodel.short_description = "pyannote segmentation-3.0: Speaker diarization. Input: 10s 16kHz mono. Output: [1, 589, 7] speaker logits."
    mlmodel.license = "MIT"

    outdir = os.path.join(os.path.dirname(__file__), "..", "sample_apps", "DiarizationDemo", "DiarizationDemo")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "SpeakerSegmentation.mlpackage")
    mlmodel.save(path)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns) / 1e6
    print(f"\nSaved to {path} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
