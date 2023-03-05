"""Voice Recognition AI project.

This script carries out the training of the Voice Recognition AI project.

@Author: Emanuel-Ionut Otel
@Company: University of Southern Denmark
@Created: 2021-09-22
@Contact: emote21@student.sdu.dk
"""

#### ---- IMPORT AREA ---- ####
import torch, torchaudio, gc, os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pod_data import PodcastDataset
from cnn import CNN_Voice
#### ---- IMPORT AREA ---- ####


#### ---- INIT AREA ---- ####
os.chdir("..")
torch.cuda.empty_cache()
BATCH_SIZE = 512
EPOCHS = 45
LEARNING_RATE = 0.0007
ANNOTATIONS_FILE = ".\data_set\Train\data_set.csv"
AUDIO_DIR = ".\data_set\Train"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 5
#### ---- INIT AREA ---- ####


def create_data_loader(train_data:PodcastDataset) -> DataLoader:
    """This function creates a data loader for the given dataset.

    :param train_data: The dataset to be used for the data loader.

    :return: The data loader.
    """
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    """This function trains the model for a single epoch.

    :param model: The model to be trained.
    :param data_loader: The data loader to be used for the training.
    :param loss_fn: The loss function to be used for the training.
    :param optimiser: The optimiser to be used for the training.
    :param device: The device to be used for the training.

    :return: The loss value for the epoch.
    """
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
    """This function trains the model for the given number of epochs.

    :param model: The model to be trained.
    :param data_loader: The data loader to be used for the training.
    :param loss_fn: The loss function to be used for the training.
    :param optimiser: The optimiser to be used for the training.
    :param device: The device to be used for the training.
    :param epochs: The number of epochs to be used for the training.
    """
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
    torch.cuda.empty_cache()

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512,
                                                           n_mels=64)

    n_fft = 256
    win_length = None
    hop_length = 256
    n_mels = 1024
    n_mfcc = 256

    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, 
                                                n_mfcc=n_mfcc, 
                                                melkwargs={'n_fft': n_fft,
                                                           'n_mels': n_mels,
                                                           'hop_length': hop_length,
                                                           'mel_scale': 'htk',})

    pd = PodcastDataset(annotations_file=ANNOTATIONS_FILE, 
                        audio_dir=AUDIO_DIR, 
                        transformation=mfcc_transform,
                        target_sample_rate=SAMPLE_RATE, 
                        device=device, 
                        num_samples=NUM_SAMPLES)

    train_dataloader = create_data_loader(pd)

    # construct model and assign it to device
    cnn = CNN_Voice().to(device)
    # print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "./model/model.pth")
    print("Trained feed forward net saved at ./model/model.pth")

    exec(open('test.py').read())
