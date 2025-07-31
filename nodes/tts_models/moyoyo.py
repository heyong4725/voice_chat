import typing
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from voice_dialogue.config.paths import TTS_MODELS_PATH
from voice_dialogue.utils.download_utils import download_file_from_huggingface
from .base import BaseTTSConfig, TTSConfigType, VoiceModelStatus


class InferenceParameters(BaseModel):
    """TTS推理参数类"""
    text_lang: str = Field(default="zh", description="文本语言")
    prompt_text: str = Field(default="", description="提示文本")
    prompt_lang: str = Field(default="zh", description="提示语言")
    top_k: int = Field(default=5, ge=1, le=100, description="Top-K采样")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-P采样")
    temperature: float = Field(default=1.0, ge=0.0, description="温度参数")
    text_split_method: str = Field(default="cut3", description="文本分割方法")
    batch_size: int = Field(default=100, ge=1, description="批处理大小")
    speed_factor: float = Field(default=1.1, ge=0.1, le=3.0, description="语速因子")
    split_bucket: bool = Field(default=True, description="是否分桶")
    return_fragment: bool = Field(default=False, description="是否返回片段")
    fragment_interval: float = Field(default=0.07, ge=0.0, description="片段间隔")
    seed: int = Field(default=233333, description="随机种子")
    # parallel_infer: bool = Field(default=False, description="是否并行推理")


class MoYoYoTTSConfig(BaseTTSConfig):
    """MoYoYo TTS配置类"""
    tts_type: TTSConfigType = TTSConfigType.MOYOYO
    repository: str
    priority: int = Field(default=1, ge=1, le=10, description="音色优先级")
    model_files: dict[str, str]
    inference_parameters: InferenceParameters

    _download_status: VoiceModelStatus = VoiceModelStatus.NOT_DOWNLOADED

    @property
    def download_status(self) -> VoiceModelStatus:
        """获取下载状态"""
        if self.is_model_complete():
            return VoiceModelStatus.DOWNLOADED
        return self._download_status

    @download_status.setter
    def download_status(self, status: VoiceModelStatus):
        """设置下载状态"""
        self._download_status = status

    def get_model_storage_path(self) -> Path:
        """获取模型存储路径"""
        storage_path = TTS_MODELS_PATH / 'moyoyo'
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def is_model_complete(self) -> bool:
        """检查模型文件是否完整"""
        storage_path = self.get_model_storage_path()
        for model_file in self.model_files.values():
            file_path = storage_path / model_file
            if not file_path.exists():
                return False
        return True

    def download_model(self, progress_callback: typing.Callable = None):
        """下载模型"""
        self.download_status = VoiceModelStatus.DOWNLOADING

        try:
            self._download_model_files(progress_callback)
            self.download_status = VoiceModelStatus.DOWNLOADED
        except Exception:
            self.download_status = VoiceModelStatus.FAILED
            raise

    def _download_model_files(self, progress_callback: typing.Callable = None):
        """从HuggingFace下载模型文件"""
        storage_path = self.get_model_storage_path()
        with ThreadPoolExecutor() as executor:
            for model_file in self.model_files.values():
                executor.submit(
                    download_file_from_huggingface,
                    storage_path,
                    self.repository,
                    model_file
                )

        if progress_callback:
            progress_callback()

    def delete_model(self):
        """删除模型核心文件"""
        storage_path = self.get_model_storage_path()
        core_files = ['gpt-weights', 'sovits-weights']
        for file_key in core_files:
            file_path = storage_path / self.model_files.get(file_key, '')
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                file_path.rmdir()
        self.download_status = VoiceModelStatus.NOT_DOWNLOADED

    # 模型文件路径属性
    @property
    def gpt_weights_path(self) -> Path:
        """GPT权重文件路径"""
        return self.get_model_storage_path() / self.model_files.get('gpt-weights', '')

    @property
    def sovits_weights_path(self) -> Path:
        """SoVITS权重文件路径"""
        return self.get_model_storage_path() / self.model_files.get('sovits-weights', '')

    @property
    def hubert_model_path(self) -> Path:
        """中文HuBERT模型路径"""
        return self.get_model_storage_path() / 'chinese-hubert-base'

    @property
    def bert_model_path(self) -> Path:
        """中文BERT模型路径"""
        return self.get_model_storage_path() / 'chinese-roberta-wwm-ext-large'

    @property
    def reference_audio_path(self) -> Path:
        """参考音频文件路径"""
        return self.get_model_storage_path() / self.model_files.get('reference_audio', '')

    def get_runtime_config(self) -> typing.Dict[str, typing.Any]:
        """获取Moyoyo运行时配置"""
        return {
            'default_v2': {
                'version': 'v2',
                'device': 'cpu',
                'is_half': False,
                't2s_weights_path': self.gpt_weights_path,
                'vits_weights_path': self.sovits_weights_path,
                'cnhuhbert_base_path': self.hubert_model_path,
                'bert_base_path': self.bert_model_path,
            }
        }
