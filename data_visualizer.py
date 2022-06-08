import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

class DataVisualizer():
    '''Takes a dataframe of emotions, ActorId, file path, Age, Sex, Race and Ethnicity 
    Visualizes the distribution of the different attributes
    Visualizes the waveforms of the different emotions'''

    def __init__(self, data) -> None:
        self.data = data
    
    def plot_data_distribution(self, attributes: list) -> None:
        attribute_mapping = {
            'Emotions': self.data.Emotion,
            'Sex': self.data.Sex,
            'Race': self.data.Race,
            'Ethicity': self.data.Ethnicity
        }

        for attribute in attributes:
            plt.title('Count of {}'.format(attribute))
            sns.countplot(x=attribute_mapping[attribute])
            plt.ylabel('Count')
            plt.xlabel('{}'.format(attribute))
            plt.show()

    def plot_waveplot_spectrogram(self) -> None:
        emotions= ['Fear', 'Happy', 'Sad', 'Angry', 'Neutral', 'Disgust']

        for emotion in emotions:
            path = np.array(self.data.Path[self.data.Emotion==emotion])[1]
            # print(path)
            data, sampling_rate = librosa.load(path)
            self.create_waveplot(data, sampling_rate, emotion)
            self.create_spectrogram(data, sampling_rate, emotion)
            plt.show()
    
    def create_waveplot(self, data, sampling_rate, emotion):
        # plt.figure(figsize=(10, 3))
        plt.title('Wave Plot: {}'.format(emotion))
        librosa.display.waveshow(data, sr=sampling_rate)

    def create_spectrogram(self, data, sampling_rate, emotion):
        # stft function converts the data into a short term fourier transform
        X = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(10, 5))
        plt.title('Spectrogram: {}'.format(emotion))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')   
        plt.colorbar()

