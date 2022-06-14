import numpy as np
import pandas as pd
import librosa
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor():
    '''Takes the dataframe containing the emotions, actor_id, sex and filenames and
       generates a dataframe that contains the extracted feature vector from each audio sample'''

    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.scaler = StandardScaler()

    def noise(self, data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def extract_features(self, data, sample_rate):
        '''Extracts the features ZCR, Chroma_stft, MFCC, RMS value and 
            Melspectrogram from the given audio sample'''
        # ZCR
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
        return result

    def get_features(self, path, augment_features=False):
        # duration and offset are used to take care of the no audio in start and the ending of each audio file.
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        
        # without augmentation
        res1 = self.extract_features(data, sample_rate)
        result = np.array(res1)
        
        if augment_features:
            # data with noise
            noise_data = self.noise(data)
            res2 = self.extract_features(noise_data, sample_rate)
            result = np.vstack((result, res2)) # stacking vertically
            
            # data with stretching and pitching
            new_data = self.stretch(data)
            data_stretch_pitch = self.pitch(new_data, sample_rate)
            res3 = self.extract_features(data_stretch_pitch, sample_rate)
            result = np.vstack((result, res3)) # stacking vertically
        
        return result

    def create_features_file(self, filename, augment_features=False):
        '''Extracts features from all audio samples and stores them in a .csv file'''

        if Path(filename).is_file():
            print("This file already exists in the current directory")
            return
        
        X, Y = [], []
        for path, emotion in zip(self.df.Path, self.df.Emotion):
            feature = self.get_features(path, augment_features)
            if augment_features:
                for ele in feature:
                    X.append(ele)
                    Y.append(emotion)
            else:
                X.append(feature)
                Y.append(emotion)   

        Features = pd.DataFrame(X)
        Features['labels'] = Y
        print("Features.shape =", Features.shape)
        Features.to_csv(filename, index=False)

    def prepare_training_data(self, filename, one_hot_encode=False):
        Features = pd.read_csv(filename)
        X = Features.iloc[: ,:-1].values
        Y = Features['labels'].values

        if one_hot_encode:
            labels = self.one_hot_encode_data(Y)
        else:
            labels = self.label_encode_data(Y)
        self.split_data(X, labels)
        self.normalize_data()
        return self.x_train, self.y_train, self.x_test, self.y_test   

    def label_encode_data(self, Y):
        le = LabelEncoder()
        le.fit(Y)
        labels = le.transform(Y)
        return labels
    
    def one_hot_encode_data(self, Y) -> np.ndarray:
        encoder = OneHotEncoder()
        labels = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        return labels

    def split_data(self, X, labels):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(X, labels, random_state=5, shuffle=True, stratify=labels)
        # print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)
    
    def normalize_data(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        # print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)