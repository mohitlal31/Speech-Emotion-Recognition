import pandas as pd
import os


class DataGenerator():
    '''This class parses all the .wav filenames and extracts details from the filename to create a dataframe'''

    def __init__(self) -> None:
        self.dataset_path = os.path.relpath('./CREMA-D/AudioWAV/')

    def create_dataframe_from_filenames(self) -> pd.DataFrame:
        '''Extract details like emotion, actor_id from the filename and return a dataframe'''

        file_list = os.listdir(self.dataset_path)
        file_emotion = []
        file_path = []
        file_actor_id = []

        for file in file_list:
            emotion = get_emotion_from_filenameEmotion(file.split('_')[2])
            file_actor_id.append(int(file.split('_')[0]))
            file_emotion.append(emotion)
            relative_path = os.path.join(self.dataset_path, file)
            file_path.append(relative_path)
            # file_path.append(os.path.abspath(relative_path))

        data = {'Emotion': file_emotion, 'Path': file_path, 'ActorID': file_actor_id}
        df = pd.DataFrame(data = data)
        demographics = pd.read_csv("./CREMA-D/VideoDemographics.csv")

        df = pd.merge(left=df, right=demographics)
        return df

def get_emotion_from_filenameEmotion(file_emotion: str) -> str:
    '''Returns the actual emotion from the 3 letter emotion value encoded in the file name'''
    if file_emotion == 'ANG':
        return 'Angry'
    elif file_emotion == 'DIS':
        return 'Disgust'
    elif file_emotion == 'SAD':
        return 'Sad'
    elif file_emotion == 'FEA':
        return 'Fear'
    elif file_emotion == 'HAP':
        return 'Happy'
    elif file_emotion == 'NEU':
        return 'Neutral'
    else:
        return 'Unknown'