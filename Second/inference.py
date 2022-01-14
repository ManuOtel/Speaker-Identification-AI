import torch
import torchaudio
import matplotlib.pyplot as plt

from cnn import CNNNetwork
from pod_data import PodcastDataset
from torch.utils.data import DataLoader

from plotcm import plot_confusion_matrix

fv = 700
tv = 128
trv = 128

n_fft = 256
win_length = None
hop_length = 256
n_mels = 256
n_mfcc = 256

class_mapping = ["both", "esben", "peter"]

ANNOTATIONS_FILE = "D:\Manu\SDU\Projects\DNN\data_set\Test\data_set.csv"
AUDIO_DIR = "D:\Manu\SDU\Projects\DNN\data_set\Test"
SAMPLE_RATE = 8000
NUM_SAMPLES = SAMPLE_RATE * 5


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    device = "cpu"

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=fv, hop_length=tv,
                                                           n_mels=trv)

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE)

    pd = PodcastDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            device,
                            NUM_SAMPLES)


    # get a sample from the urban sound dataset for inference
    input, target = pd[0][0], pd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    #predicted, expected = predict(cnn, input, target, class_mapping)
    #print(f"Predicted: '{predicted}', expected: '{expected}'")

    train_dataloader = DataLoader(pd)
    wrong = 0
    good = 0
    both = 0
    esben = 0
    peter = 0
    pp = []
    ee = []
    for input, target in train_dataloader:
        predicted, expected = predict(cnn, input, target, class_mapping)
        pp.append(class_mapping.index(predicted))
        ee.append(class_mapping.index(expected))
        if predicted == expected:
            good = good + 1
        else:
            wrong = wrong + 1
        if predicted == "both":
            both = both + 1
        if predicted == "esben":
            esben = esben + 1
        if predicted == "peter":
            peter = peter + 1
    tpp = torch.tensor(pp)
    tee = torch.tensor(ee)
    stacked = torch.stack((tpp, tee), dim=1)
    stacked[0].tolist()
    cmt = torch.zeros(3, 3, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    print(cmt)
    plt.figure(figsize=(3, 3))
    plot_confusion_matrix(cmt, class_mapping)
    plt.show()
    print("  ")
    print("  ")
    print("%%%% ---  Test Values --- %%%%")
    print("  ")
    print(f"Wrong: {wrong / 0.33}%")
    print(f"Write: {good / 0.33}%")
    print("  ")
    print(f"Both: {both}")
    print(f"Esben: {esben}")
    print(f"Peter: {peter}")
    print("  ")
    print("  ")


    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    device = "cpu"

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=fv, hop_length=tv,
                                                           n_mels=trv)

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE)

    ANNOTATIONS_FILE = "D:\Manu\SDU\Projects\DNN\data_set\Train\data_set.csv"
    AUDIO_DIR = "D:\Manu\SDU\Projects\DNN\data_set\Train"

    pd = PodcastDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            device,
                            NUM_SAMPLES)


    # get a sample from the urban sound dataset for inference
    input, target = pd[0][0], pd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    train_dataloader = DataLoader(pd)
    wrong = 0
    good = 0
    both = 0
    esben = 0
    peter = 0
    for input, target in train_dataloader:
        predicted, expected = predict(cnn, input, target, class_mapping)
        if predicted == expected:
            good = good + 1
        else:
            wrong = wrong + 1
        if predicted == "both":
            both = both + 1
        if predicted == "esben":
            esben = esben + 1
        if predicted == "peter":
            peter = peter + 1

    print("%%%% ---  Train Values --- %%%%")
    print("  ")
    print(f"Wrong: {wrong / 2.67}%")
    print(f"Write: {good / 2.67}%")
    print("  ")
    print(f"Both: {both}")
    print(f"Esben: {esben}")
    print(f"Peter: {peter}")
    print("  ")
    print("  ")