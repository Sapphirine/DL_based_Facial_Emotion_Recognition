
import os
from subprocess import PIPE, Popen

def get_emotion(filename):
    
    text_file = open(filename, 'rb')
    emotion = int(float(text_file.readline()))
    text_file.close()

    return emotion

def load_CK_emotions(root_dirname):
   
    emotions = {}

    emotion_dir = os.path.join(root_dirname, "Emotion")
    for dirname, _, file_list in os.walk(emotion_dir):
        for filename in file_list:
            record_id = filename[0:9]
            filename = os.path.join(dirname, filename)

            emotions.update({record_id : get_emotion(filename)})

    return emotions

def load_CK_videos(root_dirname):
    
    videos = {}

    for dirname, _, file_list in os.walk(root_dirname):
        for filename in file_list:
            record_id = filename[0:9]
            filename = os.path.join(dirname, filename)

            videos.update({record_id : filename})

    return videos
