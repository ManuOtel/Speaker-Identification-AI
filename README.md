# Voice-Recognition-AI
A project that aims to identify the speaker using neural networks. That would involve training a model on a dataset of audio samples based on 'Her Går Det Godt' podcast 
and their corresponding speaker labels. The model would learn to extract features from the audio that are unique to each speaker, 
such as the pitch and rhythm of their speech. Once the model is trained, it can be given a new audio sample and use the learned features 
to predict which speaker the sample belongs to. This can be useful in applications such as speaker verification for 
security purposes or identifying speakers in a conversation. The model uses convolutional neural network (CNN) applied directly on the MFCCs 
transform of the sound samples. This seems to be the best well-suited to processing sequential.

The process of transforming sound into Mel-Frequency Cepstral Coefficients (MFCCs) involves several steps. 
First, the sound wave is split into individual frames, with each frame representing a short segment of the waveform. 
Each frame is then passed through a series of filters, which separate the sound into different frequency bands. 
Next, a logarithmic compression is applied to the output of the filters, which emphasizes the lower frequencies and de-emphasizes the higher frequencies. 
Finally, a Discrete Cosine Transform (DCT) is applied to the compressed output, resulting in the MFCCs. 
These coefficients represent the unique characteristics of the sound and can be used for tasks such as speech recognition and audio classification.

Only scenario in which the model succeeded in achieving above 90% accuracy with a small training dataset was using data augumentation. 
Data augmentation is a technique used to increase the size of a dataset by generating additional data samples from the existing samples.
