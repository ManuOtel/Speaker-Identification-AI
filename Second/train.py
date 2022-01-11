import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from pod_data import PodcastDataset
from cnn import CNNNetwork

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "D:\Manu\SDU\Projects\DNN\First\data_set.csv"
AUDIO_DIR = "D:\Manu\SDU\Projects\DNN\First\dataset"
NUM_SAMPLES = 22050 * 5
SAMPLE_RATE = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    return loss.item()


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    loss_values = []
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        loss = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        loss_values.append(loss)
        print("---------------------------")
    print("Finished training")
    plt.close('All')
    plt.figure()
    plt.plot(np.array(loss_values), 'r')
    plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512,
                                                           n_mels=64)

    n_fft = 1024
    win_length = None
    hop_length = 256
    n_mels = 2048
    n_mfcc = 512

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk',
    }
    )

    pd = PodcastDataset(annotations_file=ANNOTATIONS_FILE, audio_dir=AUDIO_DIR, transformation=mfcc_transform,
                        target_sample_rate=SAMPLE_RATE, device=device, num_samples=NUM_SAMPLES)

    train_dataloader = create_data_loader(pd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    # print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
