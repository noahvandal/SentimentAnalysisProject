#! /usr/bin/env python3

from time import sleep
import threading

from sentiment_audio import AudioCNN
from sentiment_video import VideoCNN

class SentimentAnalysis():

    def __init__(self):
        """
        Initialize all the emotion score parameters,
        the CNN's, the spotify_api object,
        as well as the threads for running them in parallel.
        """
        # Global variable for the audio_emotion CNN current output
        self.audio_emotion_score = 0.0
        # Weight of the audio scores input to the final ranking
        self.audio_score_weight = 0.1
        # Global variable for the video_emotion CNN current output
        self.video_emotion_score = 0.0
        # Weight of the video scores input to the final ranking
        self.video_score_weight = 1.0 - self.audio_score_weight
        self.current_emotion_score = 0.0
        self.audio_cnn = AudioCNN()
        # TODO: Create Video CNN object
        self.video_cnn = VideoCNN()
        # TODO: Create spotify api manager object
        self.spotify_api = None
        # daemon threads clean themselves up when this script ends.
        self.audio_thread = threading.Thread(target=self.audio_inference, daemon=True)
        self.video_thread = threading.Thread(target=self.video_inference, daemon=True)

    def main(self):
        """
        Function to run which initializes all the objects and then
        runs all the threads
        """
        self.audio_thread.start()
        self.video_thread.start()
        while True:
            # if 10 seconds left until the end of the current song
            # self.spotify_api.song_almost_over()
            if False:
                self.calculate_current_emotion_score()
                # Add next song to queue based on self.current_emotion_score
            else:
                sleep(1)

    def response_filter(self, old_value, new_value):
        """
        X[n] = 0.1 * X[n - 1] + 0.9 * sample
        """
        return (0.1 * old_value) + (0.9 * new_value)

    def audio_inference(self):
        """
        Runs inference on the audio CNN and updates the current audio score
        this is done asyncronously.
        """
        while True:
            audio_emotion = self.audio_cnn.inference()
            self.audio_emotion_score = self.response_filter(self.audio_emotion_score, audio_emotion)
            sleep(1)

    def video_inference(self):
        """
        Runs inference on the video CNN and updates the current video score
        this is done asyncronously.
        """
        while True:
            video_emotion = self.video_cnn.inference()
            self.video_emotion_score = self.response_filter(self.video_emotion_score, video_emotion)
            sleep(1)

    def calculate_current_emotion_score(self):
        """
        Sets the value of self.current_emotion_score
        to the weighted sum of the video and audio emotion scores.
        """
        self.current_emotion_score = (self.audio_emotion_score * self.audio_score_weight) + (self.video_emotion_score * self.video_emotion_weight)

    def cleanup_print(self, string):
        """
        Wrapper function for appending all strings
        printed in self.cleanup
        """
        print(f"TERMINATING: {string}")

    def cleanup(self):
        """
        Any steps that need to be done to cleanup the process
        before terminating.
        """
        print("\n")
        self.cleanup_print("Shutting down AudioCNN")
        self.audio_cnn.close()
        self.cleanup_print("Shutting down VideoCNN")
        # TODO: Stop the spotify playback.
        # TODO: Something else(?)
        self.cleanup_print("Done shutting down")

# Function that actually runs when this is script is ran.
if __name__ == "__main__":
    sentiment = SentimentAnalysis()
    try:
        sentiment.main()
    except:
        sentiment.cleanup()

