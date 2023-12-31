# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ssl
import time
import warnings

import nltk
import numpy as np
import pyaudio
import scipy.signal as signal
import torch

from embeddings import SimilaritySearch

def create_unverified_https_context():
    """
    Create an unverified HTTPS context if available.

    This function checks for the availability of an unverified HTTPS context and sets it as the default HTTPS context
    if it's available. If not available, it maintains the default behavior.

    Raises
    ------
    RuntimeError
        If an error occurs during the creation of the unverified HTTPS context.

    Notes
    -----
    This functionality is primarily used to handle SSL certificate verification in HTTPS connections, allowing
    for unverified contexts when necessary.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context

    except AttributeError as e:
        raise RuntimeError("Unable to create an unverified HTTPS context.") from e

try:
    create_unverified_https_context()
    nltk.download("words")
    
    from nltk.corpus import words

except LookupError as e:
    raise LookupError("Failed to download the 'words' corpus from NLTK.") from e

from nltk.metrics.distance import jaccard_distance 
from nltk.util import ngrams

from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

warnings.filterwarnings("ignore")

class Speech2Interact():
    def __init__(self, actions:dict, wav2vec_model:str="facebook/wav2vec2-large-960h", sentence_similarity_model:str="sentence-transformers/all-mpnet-base-v2",  auto_correct:bool=True, device:str=None):
        """
        Initializes an instance of Speech2Interact with models for Wav2Vec2 and sentence similarity.

        Parameters
        ----------
        actions : dict
            Dict of lists of trigger words that might signify actions
        
        wav2vec_model : str, optional
            The name or path of the Wav2Vec2 model. Default is "facebook/wav2vec2-large-960h".

        sentence_similarity_model : str, optional
            The name or path of the sentence similarity model. Default is "sentence-transformers/all-mpnet-base-v2".

        auto_correct : bool, optional
            Performs word correction based on Jaccard distance. Default is True.

        device : str, optional
            Device to perform the encoding (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            
        Notes
        -----
        This initialization function loads and prepares models for Wav2Vec2 and sentence similarity, using 
        the specified Wav2Vec2 model and sentence similarity model to process audio and text data respectively.
        """
        if device:
            self.device = device
        
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.actions = list(actions.keys())
        self.auto_correct = auto_correct

        if self.auto_correct == True:
            self.correct_words = set(word.lower() for word in words.words() if word.isalpha())

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model)
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model)

        self.sentence_similarity_model = SimilaritySearch(
            model=sentence_similarity_model,
            device=self.device
        )

        if self.sentence_similarity_model.vectorstore is None:
            for element in actions:
                if self.device != "cpu":
                    embeddings = self.sentence_similarity_model.encode(
                        data=actions[element]
                    )
        
                    self.sentence_similarity_model.index(
                        embeddings=embeddings
                    )
            
                else:
                    self.embeddings = []
        
                    for element in actions:
                        embeddings = self.sentence_similarity_model.encode(
                            data=actions[element]
                        )
                        self.embeddings.append(embeddings)


    def stream(self, duration:int=5, sample_rate:int=16000, frames_per_buffer:int=8000, channels:int=1) -> list:
        """
        Records audio for a specified duration using the microphone input and returns the recorded frames.

        Parameters
        ----------
        duration : int, optional
            The duration of the audio recording in seconds. Default is 5 seconds.

        sample_rate : int, optional
            The sample rate of the audio. Default is 16000 Hz.

        frames_per_buffer : int, optional
            The number of frames per buffer for audio recording. Default is 8000 frames.

        channels : int, optional
            The number of audio channels. Default is 1 (mono).

        Returns
        -------
        frames : list
            A list containing the recorded audio frames.
        """
        if not all(isinstance(arg, int) and arg > 0 for arg in (duration, sample_rate, frames_per_buffer, channels)):
            raise ValueError("All parameters should be positive integers.")

        try:
            audio = pyaudio.PyAudio()

            # Set up the audio stream to record for the specified duration
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer
            )

            frames = []
            print(f"Recording for {duration} seconds...")
            
            start_time = time.time()

            while time.time() - start_time < duration:
                data = stream.read(frames_per_buffer)
                frames.append(data)

            print("Recording stopped.")

            stream.stop_stream()
            stream.close()
            audio.terminate()

            return frames

        except IOError as e:
            raise IOError(f"An error occurred with the audio stream: {e}")


    def high_pass_filter(self, waveform:np.array, sample_rate:int, cutoff_freq:float) -> np.array:
        """
        Apply a high-pass filter to the input waveform.

        Parameters
        ----------
        waveform : array_like
            Input waveform data to be filtered.

        sample_rate : int or float
            Sampling rate of the input waveform.

        cutoff_freq : float
            Cutoff frequency of the high-pass filter.

        Returns
        -------
        filtered_waveform : array_like
            The filtered waveform after applying the high-pass filter.
        """
        if not isinstance(waveform, np.ndarray):
            raise ValueError("waveform should be a numpy array.")

        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError("sample_rate should be a positive integer or float.")
        
        if not isinstance(cutoff_freq, (int, float)) or cutoff_freq <= 0:
            raise ValueError("cutoff_freq should be a positive float.")

        try:
            b, a = signal.butter(
                5,
                cutoff_freq / (0.5 * sample_rate),
                btype="highpass",
                analog=False
            )

            filtered_waveform = signal.lfilter(b, a, waveform)

        except Exception as e:
            raise RuntimeError(f"An error occurred during the filtering process: {e}")

        return filtered_waveform


    def normalize_waveform(self, frames: list) -> np.ndarray:
        """
        Normalizes waveform the waveform from audio frames.

        Parameters
        ----------
        frames : list
            List of audio frames captured from the input source.

        Returns
        -------
        waveform : np.ndarray
            Processed waveform ready for further operations.
        """
        if not frames or not all(isinstance(frame, (bytes, bytearray)) for frame in frames):
            raise ValueError("frames should be a non-empty list of bytes-like objects.")

        try:
            waveform = np.frombuffer(b''.join(frames), dtype=np.int16)
            waveform = waveform / 32768.0  # Normalize waveform to be in the range [-1.0, 1.0]
        
        except Exception as e:
            raise ValueError(f"An error occurred while processing the waveform: {e}")

        return waveform

    
    def jaccard_correction(self, sentences: str) -> str:
        """
        Performs word correction based on Jaccard distance.

        Parameters
        ----------
        sentences : str
            String containing sentences to be corrected.

        Returns
        -------
        corrected_sentence : str
            The corrected sentence after applying word corrections based on Jaccard distance.

        Notes
        -----
        This function uses Jaccard distance to correct words in the input sentences by finding similar words 
        based on the overlap of 2-grams. It takes a string of sentences and identifies words that are likely 
        misspelled or incorrect by comparing them to a list of correct words. It then replaces the incorrect 
        words with the most similar correct words, considering the Jaccard distance between their 2-gram sets.
        """
        if not isinstance(sentences, str):
            raise ValueError("Input should be a string.")

        corrected_words = []
        sentences_list = sentences.split()  # Split the string into a list of words

        for word in sentences_list:
            try:
                similar_words = [(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))), w) 
                                 for w in self.correct_words if w[0] == word[0]]

                if similar_words:
                    corrected_words.append(sorted(similar_words, key=lambda val: val[0])[0][1])

            except (KeyError, IndexError) as e:
                # Handle specific exceptions if needed
                print(f"Error processing word '{word}': {e}")
                pass

        return " ".join(corrected_words)


    def recognize_speech(self, duration:int=5, sample_rate:int=16000) -> str:
        """
        Recognizes speech from audio frames using a pre-trained model.

        Parameters
        ----------
        duration : int, optional
            The duration of the audio recording in seconds. Default is 5 seconds.

        sample_rate : int, optional
            The sample rate of the audio frames. Default is 16000 Hz.

        Returns
        -------
        transcription : str
            Transcription of the recognized speech from the input audio frames.

        Notes
        -----
        This function assumes the availability of the 'high_pass_filter' function, a model, and a processor
        to perform speech recognition. Adjustments or connections to these components need to be ensured
        for the proper functionality of this function.
        """
        try:
            frames = self.stream(
                duration=duration
            )

            waveform = self.normalize_waveform(
                frames=frames
            ) 

            waveform = self.high_pass_filter(
                waveform=waveform, 
                sample_rate=sample_rate, 
                cutoff_freq=300
            )

            waveform_tensor = torch.from_numpy(waveform).float()

            # Perform speech recognition
            input_values = self.processor(
                waveform_tensor, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_values
            
            with torch.no_grad():  # Ensure that we're not computing gradients
                logits = self.wav2vec_model(input_values).logits

            predicted_ids = torch.argmax(
                logits, 
                dim=-1
            )

            transcription = self.processor.batch_decode(predicted_ids)[0]
            transcription = transcription.lower()

            if transcription and self.auto_correct == True:
                corrected_sentences = self.jaccard_correction(
                    sentences=transcription
                )

            else:
                raise RuntimeError("Missing components: transcription.")

            # Check for empty embeddings or model issues
            if not self.embeddings or not self.sentence_similarity_model or not self.actions:
                raise RuntimeError("Missing components: embeddings, sentence_similarity_model, or actions.")

            target = self.sentence_similarity_model.encode(
                data=corrected_sentences
            )

            if self.device == "cpu":
                index = self.sentence_similarity_model.cosine(
                    embeddings=self.embeddings,
                    target=target
                )
            else:
                raise RuntimeError("Invalid device. Only 'cpu' is supported.")

            return self.actions[index]

        except Exception as e:
            raise RuntimeError(f"An error occurred during speech recognition: {e}")


# Example usage:
if __name__ == "__main__":
    actions = {
        "validate": [
            "confirm", "approve", "verify", "validate", 
            "authenticate", "ratify", "endorse", "certify", 
            "pass", "authorize", "accredit", "yes"
        ],
        "invalidate": [
            "reject", "deny", "invalidate", "disapprove", 
            "refuse", "void", "nullify", "revoke", 
            "discredit", "disqualify", "abrogate", "annul", "no"
        ]
    }

    instance = Speech2Interact(
        actions=actions,
        wav2vec_model="facebook/wav2vec2-large-960h", 
        sentence_similarity_model="sentence-transformers/all-mpnet-base-v2"
    )

    while True:
        action = instance.recognize_speech(
            duration=3
        )

        print(action)
