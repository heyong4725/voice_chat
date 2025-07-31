import typing
from pathlib import Path

from pydantic import BaseModel, Field

from voice_dialogue.config import paths
from .base import BaseTTSConfig, TTSConfigType


class InferenceParameters(BaseModel):
    """Kokoro TTS 推理参数"""
    voice: str = Field(description="语音角色名称")
    speed: float = Field(default=1.0, description="语音播放速度")
    is_phonemes: bool = Field(default=True, description="是否使用音素")


class ModelFiles(BaseModel):
    """模型文件配置"""
    model: str = Field(default='', description="模型文件名")
    voice: str = Field(default='', description="语音文件名")
    vocab_config: str = Field(default=None, description="音素配置文件名")


class KokoroTTSConfig(BaseTTSConfig):
    tts_type: TTSConfigType = TTSConfigType.KOKORO
    inference_parameters: InferenceParameters
    model_files: ModelFiles

    def get_model_storage_path(self) -> Path:
        storage_path = paths.TTS_MODELS_PATH / 'kokoro'
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def is_model_complete(self) -> bool:
        storage_path = self.get_model_storage_path()
        for model_file in self.model_files.model_dump().values():
            if not model_file:
                continue

            file_path = storage_path / model_file
            if not file_path.exists():
                return False
        return True

    def download_model(self, progress_callback: typing.Callable = None):
        pass

    def delete_model(self):
        pass

    @property
    def model_path(self):
        return self.get_model_storage_path() / self.model_files.model

    @property
    def voices_path(self):
        return self.get_model_storage_path() / self.model_files.voice

    @property
    def vocab_config_path(self):
        return self.get_model_storage_path() / self.model_files.vocab_config
