import pyarrow as pa
from dora import Node, DoraStatus

# Import necessary components from the original project
from tts_manager import tts_manager
from speaker_config import get_tts_config_by_speaker_name

class TTSNode:
    """
    A dora node that synthesizes text into audio using a TTS manager.
    """

    def __init__(self):
        # Initialize the TTS client. We'll use a default speaker.
        # This could be configured via environment variables in the YAML.
        import os
        speaker_name = os.environ.get("TTS_SPEAKER", "kokoro") # Default to 'kokoro' if not set
        tts_config = get_tts_config_by_speaker_name(speaker_name)
        if not tts_config:
            raise ValueError(f"Could not find TTS config for speaker: {speaker_name}")

        self.tts_instance = tts_manager.create_tts(tts_config)
        self.tts_instance.setup()
        self.tts_instance.warmup()
        print(f"TTS instance for speaker '{speaker_name}' warmed up and ready.")

    def __call__(self, event, dora_node):
        if event["type"] == "INPUT":
            if event["id"] == "text_to_synthesize":
                sentence = event["value"][0].as_py()
                print(f'TTS Node received text: "{sentence}"')

                try:
                    # Synthesize the audio
                    audio_waveform = self.tts_instance.synthesize(sentence)
                    
                    # The output is likely a numpy array. We need to convert it to bytes.
                    audio_bytes = audio_waveform.tobytes()

                    # Send the synthesized audio data to the next node
                    dora_node.send_output("synthesized_audio", pa.array([audio_bytes]))

                except Exception as e:
                    print(f"Error during TTS synthesis: {e}")

        return DoraStatus.CONTINUE

    def __del__(self):
        print("TTS node shutting down.")

if __name__ == "__main__":
    node = Node()
    tts_node = TTSNode()

    for event in node:
        status = tts_node(event, node)
        if status == DoraStatus.STOP:
            break