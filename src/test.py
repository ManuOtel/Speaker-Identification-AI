"""Voice Recognition AI project.

This script carries out the training of the Voice Recognition AI project.

@Author: Emanuel-Ionut Otel
@Company: University of Southern Denmark
@Created: 2021-09-22
@Contact: emote21@student.sdu.dk
"""
#### ---- IMPORT AREA ---- ####
import torch, torchaudio
from cnn import CNN_Voice
from pod_data import PodcastDataset
from torch.utils.data import DataLoader
#### ---- IMPORT AREA ---- ####


#### ---- INIT AREA ---- ####
os.chdir("..")
class_mapping = ["both", "esben", "peter"]
ANNOTATIONS_FILE = ".\data_set\Test\data_set.csv"
AUDIO_DIR = ".\data_set\Test"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 5
#### ---- INIT AREA ---- ####


def predict(model, input, target, class_mapping):
    """This function makes a prediction for the given input.

    :param model: The model to be used for the prediction.
    :param input: The input to be used for the prediction.
    :param target: The target to be used for the prediction.
    :param class_mapping: The class mapping to be used for the prediction.

    :return: The predicted and expected values.
    """
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
    cnn = CNN_Voice()
    state_dict = torch.load("./models/model.pth")
    cnn.load_state_dict(state_dict)

    device = "cpu"

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    n_fft = 256
    win_length = None
    hop_length = 256
    n_mels = 1024
    n_mfcc = 256

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk',
    })

    pd = PodcastDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mfcc_transform,
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
    cnn = CNN_Voice()
    state_dict = torch.load("./models/model.pth")
    cnn.load_state_dict(state_dict)

    device = "cpu"

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    n_fft = 256
    win_length = None
    hop_length = 256
    n_mels = 1024
    n_mfcc = 256

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk',
    })

    ANNOTATIONS_FILE = "D:\Manu\SDU\Projects\DNN\data_set\Train\data_set.csv"
    AUDIO_DIR = "D:\Manu\SDU\Projects\DNN\data_set\Train"

    pd = PodcastDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mfcc_transform,
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