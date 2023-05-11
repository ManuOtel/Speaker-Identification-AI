# Speaker Identification AI

This repository contains the code for a university project based on developing a Speaker Identification AI using very limited data. The main target was to train a simple CNN for recognising the speaker in each data sample. The data come from the podcast 'Her Går Det Godt', and the system was designed to identify one or both hosts when they are speaking, based on 5 second samples. 

The project was conducted at the University of Southern Denmark, for the Deep Neural Networks course.

## Requirements

To run the code, you will need the following:

- Python 3.x
- PyTorch 1.x

## Installation

1. Clone this repository:

```bash
git clone https://github.com/ManuOtel/Voice-Recognition-AI
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Before starting the training/testing make sure to extract the data set from the .rar archive inside ./dataset folder. 

 - Training script:
```bash
cd src
python train.py
```

 - Testing script:
```bash
cd src
python test.py
```

## Reference

[1] J. Martinez, H. Perez, E. Escamilla, and M. M. Suzuki, "Speaker recognition using Mel Frequency Cepstral Coefficients (MFCC) and Vector quantization (VQ) techniques," CONIELECOMP 2012, 22nd International Conference on Electrical Communications and Computers, Cholula, Puebla, Mexico, 2012, [link](https://ieeexplore.ieee.org/document/6189918)



## Contact

For further discussions, ideas, or collaborations please contact: [emote21@student.sdu.dk](mailto:emote21@student.sdu.dk)
