import typing

import numpy as np
from pywhispercpp.model import Model

from voice_dialogue.config import paths
from voice_dialogue.services.speech.recognizers.manager import asr_tables
from voice_dialogue.services.speech.recognizers.models.base import ASRInterface
from voice_dialogue.services.speech.recognizers.utils import ensure_minimum_audio_duration
from voice_dialogue.utils.logger import logger


@asr_tables.register('asr_classes', 'whisper')
class WhisperCppClient(ASRInterface):
    """Whisper C++ API客户端"""
    supported_langs = ['en', 'zh', ]

    def __init__(self):
        super().__init__()
        self.whisper: typing.Optional[Model] = None
        self.language = "en"

    def setup(self, **kwargs) -> None:
        model = kwargs.get('model', 'medium')
        if model == "medium":
            model = "medium-q5_0"
        else:
            model = "large-v3-turbo-q5_0"

        models_dir = paths.ASR_MODELS_PATH / "whisper"
        self.whisper = Model(model=model, models_dir=models_dir)

    def warmup(self) -> None:
        logger.info('[INFO] Warming up Whisper model...')
        try:
            self.transcribe(self.warmup_audiodata)
            logger.info('[INFO] Whisper model warmed up.')
        except Exception as e:
            logger.warning(f'[WARNING] Whisper model warmup failed: {e}')

    def transcribe(self, audio_array: np.ndarray, language="en"):
        if language == "zh":
            prompt = "以下是简体中文普通话的句子。"
        else:
            prompt = "The following is an English sentence."

        audio_array = ensure_minimum_audio_duration(audio_array)

        # print('............... language:', language)
        segments = self.whisper.transcribe(
            audio_array, language=language, initial_prompt=prompt, print_progress=False
        )
        text = []
        for segment in segments:
            content = segment.text
            # if not content.endswith(()):
            # content += ','
            text.append(content)
        text = " ".join(text)
        return text
