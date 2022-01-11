import os

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython

matplotlib.rcParams['figure.figsize'] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SPEECH_FILE = "_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
  os.makedirs('_assets', exist_ok=True)
  with open(SPEECH_FILE, 'wb') as file:
    file.write(requests.get(SPEECH_URL).content)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)

IPython.display.Audio(SPEECH_FILE)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
  waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
  emission, _ = model(waveform)

class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, ignore):
    super().__init__()
    self.labels = labels
    self.ignore = ignore

  def forward(self, emission: torch.Tensor) -> str:
    """Given a sequence emission over labels, get the best path string
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

    Returns:
      str: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [i for i in indices if i not in self.ignore]
    return ''.join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(
    labels=bundle.get_labels(),
    ignore=(0, 1, 2, 3),
)
transcript = decoder(emission[0])

print(transcript)