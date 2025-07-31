import re
import typing

import numpy as np
from funasr_onnx import SeacoParaformer, CT_Transformer

from voice_dialogue.config import paths
from voice_dialogue.services.speech.recognizers.manager import asr_tables
from voice_dialogue.services.speech.recognizers.models.base import ASRInterface
from voice_dialogue.services.speech.recognizers.utils import ensure_minimum_audio_duration
from voice_dialogue.utils.logger import logger


@asr_tables.register('asr_classes', 'funasr')
class FunASRClient(ASRInterface):
    """FunASR API客户端"""
    supported_langs = ['zh']

    def __init__(self):
        super().__init__()
        self.funasr_model: typing.Optional[SeacoParaformer] = None
        self.punc_model: typing.Optional[CT_Transformer] = None

    def setup(self, **kwargs) -> None:
        # 设置模型缓存目录
        models_dir = paths.ASR_MODELS_PATH / "funasr"
        asr_model_path = models_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        punc_model_path = models_dir / "punc_ct-transformer_cn-en-common-vocab471067-large"
        self.funasr_model = SeacoParaformer(asr_model_path, quantize=True)
        self.punc_model = CT_Transformer(punc_model_path, quantize=True)

    def warmup(self) -> None:
        logger.info('[INFO] Warming up FunASR model...')
        try:
            self.transcribe(self.warmup_audiodata)
            logger.info('[INFO] FunASR model warmed up.')
        except Exception as e:
            logger.warning(f'[WARNING] FunASR model warmup failed: {e}')

    def _fix_spaced_uppercase(self, text: str) -> str:
        """
        修复类似 " G N O M E " 这样的大写字母间有空格的字符串，将其替换为 "GNOME"
        """
        # 匹配大写字母之间的空格模式，至少2个大写字母
        pattern = r'([A-Z])\s+([A-Z](?:\s+[A-Z])*)'

        def replace_func(match):
            # 移除所有空格
            return match.group(0).replace(' ', '')

        return re.sub(pattern, replace_func, text)

    def transcribe(self, audio_array: np.ndarray, language="auto"):
        audio_array = ensure_minimum_audio_duration(audio_array)

        segments = self.funasr_model(wav_content=audio_array, hotwords='')

        transcibed_texts = []
        for segment in segments:
            content = segment.get("preds", "")
            try:
                content, _ = self.punc_model(content)
            except UnboundLocalError as e:
                logger.warning(f'[WARNING] Punctuation model failed: {e}')
            content = self._fix_spaced_uppercase(content)
            transcibed_texts.append(content)
        return " ".join(transcibed_texts)
