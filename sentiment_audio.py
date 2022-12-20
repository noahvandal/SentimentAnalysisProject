#!/usr/bin/env python3

import os
import pyaudio
import librosa
import wave
import random
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import model_from_json

# Default values for PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class Microphone():
    def __init__(self, rate=RATE, chunk=CHUNK):
        """
        Initialize all the values for using PyAudio
        correctly.
        """
        self.mic = None
        self.audio = None
        self.chunk = chunk
        self.rate = rate
        # We write the file into ram instead of 
        # wasting time writing it to disk.
        self.wav_file = "/dev/shm/tmp_out.wav"
        self.open()

    def open(self):
        """
        Opens the audio stream from the default device.
        """
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        # Open a stream from the default device
        self.mic = self.audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    def close(self):
        """
        Handles cleaning up all the objects and streams
        when we are done with them as well as the temp file
        used for getting the data into something librosa can
        read easily (.wav) file.
        """ 
        self.mic.close()
        self.audio.terminate()
        self.mic = None
        self.audio = None
        try:
            os.remove(self.wav_file)
        except OSError:
            pass
    
    def _mic_to_wav_file(self, seconds):
        """
        Converts a `seconds` long clip of audio from the
        default microphone to a .wav file for processing by
        librosa.
        """
        data = []
        if self.mic is not None:
            # https://www.youtube.com/watch?v=SlL7VYYaTGA
            for idx in range(int(RATE/CHUNK * seconds)):
                data.append(self.mic.read(CHUNK))
            with wave.open(self.wav_file, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(data))
        else:
            print("Microphone not opened.")

    def get_sample(self, seconds):
        """
        Public function which returns an array of the last
        <seconds> samples at <framerate>.
        """
        self._mic_to_wav_file(seconds)
        data, framerate = librosa.load(self.wav_file)
        return (data, framerate)

class AudioCNN():

    def __init__(self):
        """
        Initialize the microphone and any other needed objects.
        """
        self.mic = Microphone()

    def noise(self,data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self,data): ## rate = 0.8
        return librosa.effects.time_stretch(data,rate=0.8)

    def shift(self,data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self,data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data,sr=sampling_rate,n_steps=1)

    def extract_features_vanilla(self, data, sample_rate):
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

    def get_features_vanilla_datafile(self,data,sample_rate):
        # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
        # without augmentation

        res1 = self.extract_features_vanilla(data,sample_rate)
        result = np.array(res1)
        
        # data with noise
        noise_data = self.noise(data)
        res2 = self.extract_features_vanilla(noise_data,sample_rate)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sample_rate)
        res3 = self.extract_features_vanilla(data_stretch_pitch,sample_rate)
        result = np.vstack((result, res3)) # stacking vertically
        
        return result
    
    def emotion_score_from_sound(self,sound_data,sample_rate):  ## input the sound data as a two channel numpy array with raw audio data

        ## this is a local file, so ensure that these parameters are changed if filepath is modified
        abs_path = "C:/Users/noahv/OneDrive/NDSU Research/Coding Projects/ML 677 Project Testings/sentiment-analysis/"

        filename = abs_path + "CNN_audio_model_v1.h5"
        model = model_from_json(open(abs_path + "CNN_audio_model_v1.json", "r").read())
        model.load_weights(filename)
        
        ## emotion list data for output
        emotion_list = ['neutral','calm','happy','sad','angry','fear','disgust','surprise']
        emotion_value = [5,5,10,0,3,3,3,7] ## happy = 10, sad = 0, neutral = calm = 5; disgust = angry = fear = 3; surprise = 7
        
        # sound_data = data[:,0]
        sound_features = self.get_features_vanilla_datafile(sound_data,sample_rate)
        
        emotion_pred = model.predict(sound_features)
        
        ## getting total prediction from each of the output feature vectors
        reg_sound = emotion_pred[0]
        noise_sound = emotion_pred[1]
        stretch_sound = emotion_pred[2]
        total_sound = (reg_sound + noise_sound + stretch_sound)/3
        
        ## emotion score of entire array
        score = total_sound * emotion_value
        score = np.sum(score)

        ## getting the actual predicted emotion output
        loc_emotion = np.where(total_sound > 0.75)
        loc_emotion = loc_emotion[0][0]
        loc_emotion = emotion_list[loc_emotion]
        
        return score  ## can also return loc_emotion if desired

    def close(self):
        """
        Cleanup any leftover objects like the microphone.
        """
        self.mic.close()

    def inference(self,recorded_sound,sampling_rate):
        """
        Run the actual inference engine.
        """
        # TODO: implement AudioCNN inference function

        sound_score = self.emotion_score_from_sound(recorded_sound,sampling_rate)

        out = random.randint(0,10)
        return sound_score

# # C:\Users\noahv\OneDrive\NDSU Research\Coding Projects\ML 677 Project Testings\Speech Databases\RAVDESS\Audio_Speech_Actors_01-24
# abs_path = "C:/Users/noahv/OneDrive/NDSU Research/Coding Projects/ML 677 Project Testings/Speech Databases/RAVDESS/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"


# test = AudioCNN()

# audio, sr = librosa.load(abs_path)


# score = test.inference(audio,sr)

# print(score)

