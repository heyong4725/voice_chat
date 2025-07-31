import numpy as np
import pyarrow as pa
from dora import Node, DoraStatus

# We will need to import the ASR manager from the original project.
# This assumes that the `src` directory is in the Python path.
# In a real-world refactoring, we might move these shared utilities
# into a common library.
from voice_dialogue.services.speech.recognizers import asr_manager

class ASRNode:
    """
    A dora node that transcribes audio chunks using the ASR manager.
    """

    def __init__(self):
        # Initialize the ASR client. We'll default to English for now.
        # This could be configured via environment variables in the YAML.
        import os
        asr_language = os.environ.get("ASR_LANGUAGE", "en") # Default to 'en' if not set
        self.asr_client = asr_manager.create_asr(asr_language)
        print(f"ASR client initialized with language: {asr_language}")
        self.asr_client.setup()
        self.asr_client.warmup()
        print("ASR client warmed up and ready.")

    def __call__(self, event, dora_node):
        if event["type"] == "INPUT":
            if event["id"] == "voice_utterance":
                # The input is a pyarrow array containing the binary audio data.
                audio_data = event["value"][0].as_py()

                # Convert the raw bytes to a numpy array, which the ASR client expects.
                # The original used int16, so we'll assume that format.
                audio_np = np.frombuffer(audio_data, dtype=np.int16)

                # Transcribe the audio.
                transcribed_text = self.asr_client.transcribe(audio_np)

                if transcribed_text and transcribed_text.strip():
                    print(f'Transcribed Text: "{transcribed_text}"')
                    # Send the transcribed text to the next node.
                    dora_node.send_output("transcribed_text", pa.array([transcribed_text]))

        return DoraStatus.CONTINUE

    def __del__(self):
        # Clean up resources if necessary.
        print("ASR node shutting down.")

if __name__ == "__main__":
    node = Node()
    asr_node = ASRNode()

    for event in node:
        status = asr_node(event, node)
        if status == DoraStatus.STOP:
            break