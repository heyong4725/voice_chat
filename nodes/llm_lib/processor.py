import pathlib
import typing

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
)
from langchain_core.runnables import RunnableWithMessageHistory

from voice_dialogue.utils.logger import logger


def create_langchain_chat_llamacpp_instance(
        local_model_path: str,
        model_params: dict | None = None
) -> ChatLlamaCpp:
    logger.info(">>>>>>> Initializing LlamaCpp Langchain instance...")

    model_path = pathlib.Path(local_model_path)
    llamacpp_langchain_instance = ChatLlamaCpp(
        model_path=str(model_path),
        **model_params
    )

    return llamacpp_langchain_instance


def create_langchain_pipeline(langchain_instance, system_prompt: str, get_session_history: typing.Callable):
    prompt = ChatPromptTemplate(messages=[
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    langchain_pipeline = prompt | langchain_instance
    if get_session_history is None:
        raise NotImplementedError
    chain_with_history = RunnableWithMessageHistory(langchain_pipeline, get_session_history,
                                                    history_messages_key='history')
    return chain_with_history


def warmup_langchain_pipeline(pipeline):
    logger.info("Warmup chat pipeline...")

    user_input = 'Hello, this is warming up step, if you understand, output "Ok".'
    config = {"configurable": {"session_id": 'warmup'}}
    for _ in pipeline.stream(input={'input': user_input}, config=config):
        pass


def preprocess_sentence_text(sentences):
    sentence_text = ''.join(sentences)
    if sentence_text:
        sentence_mark = sentence_text[-1]
        sentence_content = sentence_text[:-1].replace('!', ',').replace('?', ',').replace('.', ',')
        sentence_text = f'{sentence_content}{sentence_mark}'
    return sentence_text
