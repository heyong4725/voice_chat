import ctypes
import time
import pyarrow as pa
from dora import Node, DoraStatus

# Assuming LIBRARIES_PATH is correctly set in the execution environment
# In a real scenario, you might need a more robust way to locate this.
LIBRARIES_PATH = "../assets/libraries"

class AudioCaptureNode:
    """
    A dora node that captures audio using a custom macOS library with AEC.
    """

    def __init__(self):
        self.audio_recorder = None
        try:
            self.audio_recorder = ctypes.CDLL(f"{LIBRARIES_PATH}/libAudioCapture.dylib")
            # Define function signatures for type safety
            self.audio_recorder.getAudioData.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_bool)]
            self.audio_recorder.getAudioData.restype = ctypes.POINTER(ctypes.c_ubyte)
            self.audio_recorder.freeAudioData.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
            self.audio_recorder.startRecord.restype = None
            self.audio_recorder.stopRecord.restype = None
        except Exception as e:
            print(f"Error loading libAudioCapture.dylib: {e}")
            self.audio_recorder = None

    def __call__(self, event, dora_node):
        if not self.audio_recorder:
            print("Audio recorder library not loaded. Cannot capture audio.")
            return DoraStatus.STOP

        if event["type"] == "INPUT":
            if event["id"] == "tick":
                size = ctypes.c_int(0)
                is_voice_active = ctypes.c_bool(False)
                
                data_ptr = self.audio_recorder.getAudioData(ctypes.byref(size), ctypes.byref(is_voice_active))

                if data_ptr and size.value > 0:
                    audio_data = bytes(data_ptr[:size.value])
                    
                    # Send a structured message with audio and the VAD flag
                    output = {
                        "audio": pa.array([audio_data]),
                        "is_voice_active": pa.array([is_voice_active.value])
                    }
                    dora_node.send_output("audio_chunk", pa.StructArray.from_arrays(output.values(), names=output.keys()))

                    # Free the memory allocated by the native library
                    self.audio_recorder.freeAudioData(data_ptr)
                
        return DoraStatus.CONTINUE

    def __del__(self):
        # Ensure the recording is stopped when the node is destroyed
        if self.audio_recorder:
            print("Stopping audio recording...")
            self.audio_recorder.stopRecord()

if __name__ == "__main__":
    node = Node()
    audio_capture = AudioCaptureNode()
    
    # Start recording when the node is initialized
    if audio_capture.audio_recorder:
        print("Starting audio recording...")
        audio_capture.audio_recorder.startRecord()

    for event in node:
        status = audio_capture(event, node)
        if status == DoraStatus.STOP:
            break