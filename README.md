# BALanced Execution through Natural Activation : a human-computer interaction methodology for code running.
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

BALENA is a voice interaction framework utilizing state-of-the-art natural language processing and audio processing models to create a system that can interpret voice commands and associate them with predefined actions. The framework leverages the power of transformers and signal processing to understand user intent via spoken language and correlates them with a series of predefined actionable responses.

![Plot](https://github.com/louisbrulenaudet/balena/blob/main/thumbnail.png?raw=true)

## Features
- **Real-time audio streaming and recording**: Record audio from the microphone in real time for processing.
- **Speech recognition with Wav2Vec 2.0**: Use a pre-trained Wav2Vec 2.0 model to convert speech to text.
- **Text similarity and action triggering**: Encode the transcribed text to a vector space and find the closest action using sentence similarity techniques.
- **High-pass filtering**: Process the audio signal with a high-pass filter to enhance signal quality.
- **Auto-correction**: Utilize the Jaccard distance to correct words in the transcribed text auto-magically.
- **Framework flexibility**: Support for different device execution contexts, allowing for usage on both CPU and CUDA devices.

## Dependencies

Below is a list of the main dependencies for Speech2Interact:
- `ssl`: To handle SSL/TLS encryption for establishing secure connections.
- `time`: For timing operations, such as streaming duration.
- `warnings`: To manage runtime warnings produced during execution.
- `nltk`: Utilized for natural language processing tasks like Jaccard distance calculations.
- `numpy`: Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- `pyaudio`: Allows for real-time audio recording and playback.
- `scipy`: Used for signal processing, including applying the high-pass filter.
- `torch`: An open-source machine learning framework that accelerates the path from research prototyping to production deployment.
- `transformers`: Provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, and more.
- `sentence_transformers`: A Python framework for state-of-the-art sentence and text embeddings.

## Pre-trained Models

- **Wav2Vec 2.0**: `facebook/wav2vec2-large-960h`
- **Sentence Similarity**: `sentence-transformers/all-mpnet-base-v2`

## Usage
Here's how you can use `apple-ocr`:

1. **Installation**: Install the required libraries, including `Torch`, `NumPy`...
2. **Initialization**: Create an instance of the `OCR` class, providing an image to be processed.

```python
from balena.hci import Speech2Interact

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

action = instance.recognize_speech(
		duration=3
)
```
The `recognize_speech` function captures audio for a set duration, processes the audio, and attempts to match the spoken words with a predefined action.

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2023,
	author = {Louis Brulé Naudet},
	title = {BALanced Execution through Natural Activation : a human-computer interaction methodology for code running},
	howpublished = {\url{https://github.com/louisbrulenaudet/balena}},
	year = {2023}
}

```
## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).