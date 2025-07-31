
import pyarrow as pa
from dora import Node, DoraStatus
import time
import numpy as np

class SpeechStateMonitorNode:
    """
    A dora node that monitors voice activity and buffers audio chunks 
    into a complete utterance.
    """

    def __init__(self):
        self.audio_buffer = []
        self.is_speaking = False
        self.last_speech_time = None
        self.silence_threshold_ms = 500  # 0.5 seconds of silence to send
        print("Speech state monitor initialized.")

    def __call__(self, event, dora_node):
        if event["type"] == "INPUT":
            if event["id"] == "audio_chunk":
                # The input is a structured pyarrow array
                data = event["value"][0].as_py()
                audio_data = data["audio"]
                is_voice_active = data["is_voice_active"]

                if is_voice_active:
                    self.is_speaking = True
                    self.last_speech_time = time.time()
                    self.audio_buffer.append(audio_data)
                
                elif self.is_speaking and not is_voice_active:
                    # User was speaking, but now there's silence.
                    # Check if the silence duration has passed the threshold.
                    if self.last_speech_time and (time.time() - self.last_speech_time) * 1000 > self.silence_threshold_ms:
                        print("Silence threshold reached. Sending full utterance.")
                        # Concatenate all buffered audio chunks
                        full_utterance = np.concatenate([np.frombuffer(chunk, dtype=np.int16) for chunk in self.audio_buffer])
                        
                        # Send the complete utterance as a single audio chunk
                        dora_node.send_output("voice_utterance", pa.array([full_utterance.tobytes()]))
                        
                        # Reset the state for the next utterance
                        self.audio_buffer = []
                        self.is_speaking = False
                        self.last_speech_time = None

        return DoraStatus.CONTINUE

    def __del__(self):
        print("Speech state monitor node shutting down.")

if __name__ == "__main__":
    node = Node()
    monitor_node = SpeechStateMonitorNode()

    for event in node:
        status = monitor_node(event, node)
        if status == DoraStatus.STOP:
            break
