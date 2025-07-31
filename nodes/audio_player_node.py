import pyarrow as pa
from dora import Node, DoraStatus
import soundfile as sf
from playsound import playsound
import tempfile
import numpy as np

class AudioPlayerNode:
    """
    A dora node that plays synthesized audio chunks.
    """

    def __init__(self):
        # No special initialization needed for playsound
        print("Audio player node initialized.")

    def __call__(self, event, dora_node):
        if event["type"] == "INPUT":
            if event["id"] == "synthesized_audio":
                audio_bytes = event["value"][0].as_py()
                print(f"Audio Player received {len(audio_bytes)} bytes of audio data.")

                try:
                    # The audio data is in bytes, convert it back to a numpy array.
                    # We assume int16 format, as that's common for TTS.
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

                    # Use a temporary file to play the audio
                    with tempfile.NamedTemporaryFile('w+b', suffix='.wav') as soundfile:
                        # The sample rate should ideally be passed along with the audio data.
                        # We'll assume 16000 for now, as it's a common standard.
                        sample_rate = 16000
                        sf.write(soundfile, audio_np, samplerate=sample_rate, subtype='PCM_16', closefd=False)
                        playsound(soundfile.name, block=True)
                        print("Finished playing audio.")

                except Exception as e:
                    print(f"Error playing audio: {e}")

        return DoraStatus.CONTINUE

    def __del__(self):
        print("Audio player node shutting down.")

if __name__ == "__main__":
    node = Node()
    player_node = AudioPlayerNode()

    for event in node:
        status = player_node(event, node)
        if status == DoraStatus.STOP:
            break