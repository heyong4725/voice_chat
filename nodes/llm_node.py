import pyarrow as pa
from dora import Node, DoraStatus
import unicodedata

# Import necessary components from the original project and langchain
from voice_dialogue.config import paths
from voice_dialogue.config.llm_config import get_llm_model_params
from voice_dialogue.config.user_config import get_prompt
from voice_dialogue.services.text.processor import (
    preprocess_sentence_text, create_langchain_chat_llamacpp_instance,
    create_langchain_pipeline, warmup_langchain_pipeline
)
from langchain_core.chat_history import InMemoryChatMessageHistory

# A simple in-memory cache for conversation history, similar to the original.
chat_history_cache = {}

class LLMNode:
    """
    A dora node that generates text responses using a LangChain pipeline.
    """

    def __init__(self):
        # --- 1. Initialize the LangChain Model and Pipeline ---
        model_path = paths.LLM_MODELS_PATH / 'qwen' / 'Qwen3-8B-Q6_K.gguf'
        model_params = get_llm_model_params()
        
        self.model_instance = create_langchain_chat_llamacpp_instance(
            local_model_path=model_path, model_params=model_params
        )
        
        # The get_session_history function is now a method of this class
        prompt = get_prompt("en") # Default to English
        self.pipeline = create_langchain_pipeline(self.model_instance, prompt, self.get_session_history)
        
        warmup_langchain_pipeline(self.pipeline)
        print("LLM Pipeline warmed up and ready.")

        # --- 2. Sentence Splitting Logic (copied from original) ---
        self.english_sentence_end_marks = {'!', '?', '.', ',', ':', ';'}
        self.chinese_sentence_end_marks = {'，', '。', '！', '？', '：', '；', '、'}
        self.sentence_end_marks = self.english_sentence_end_marks | self.chinese_sentence_end_marks

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retrieves the chat history for a given session ID from our cache."""
        if session_id not in chat_history_cache:
            chat_history_cache[session_id] = InMemoryChatMessageHistory()
        return chat_history_cache[session_id]

    def __call__(self, event, dora_node):
        if event["type"] == "INPUT":
            if event["id"] == "user_question":
                user_question = event["value"][0].as_py()
                print(f'LLM Node received question: "{user_question}"')

                # A unique session ID for this conversation turn.
                # In a real app, this would be managed more robustly.
                session_id = "dora_session"
                config = {"configurable": {"session_id": session_id}}

                chunks = []
                is_first_sentence = True

                try:
                    for chunk in self.pipeline.stream(input={'input': user_question}, config=config):
                        if not chunk.content or chunk.content in {'<think>', '\n\n', '</think>'}:
                            continue

                        before_punct, sentence_end_mark, remain_content = self._process_chunk_content(chunk.content)
                        if before_punct: chunks.append(before_punct)
                        if sentence_end_mark: chunks.append(sentence_end_mark)

                        sentence = preprocess_sentence_text(chunks)
                        if not sentence: 
                            chunks.append(remain_content)
                            continue

                        if self._should_end_sentence(sentence, sentence_end_mark, is_first_sentence):
                            print(f"Sending sentence: {sentence}")
                            dora_node.send_output("llm_response", pa.array([sentence]))
                            chunks = [remain_content] if remain_content else []
                            is_first_sentence = False
                        else:
                            if remain_content: chunks.append(remain_content)

                    # Handle any remaining text
                    if chunks:
                        remaining_sentence = preprocess_sentence_text(chunks)
                        if remaining_sentence and remaining_sentence.strip() not in self.sentence_end_marks:
                            print(f"Sending remaining sentence: {remaining_sentence}")
                            dora_node.send_output("llm_response", pa.array([remaining_sentence]))

                except Exception as e:
                    print(f"Error in LLM pipeline stream: {e}")

        return DoraStatus.CONTINUE

    # --- Helper methods for sentence splitting (copied from original) ---
    def _is_punctuation(self, char: str) -> bool:
        return unicodedata.category(char).startswith('P') if char and len(char) == 1 else False

    def _process_chunk_content(self, chunk_content: str) -> tuple:
        if not chunk_content: return '', '', ''
        for i in range(len(chunk_content) - 1, -1, -1):
            char = chunk_content[i]
            if self._is_punctuation(char):
                return chunk_content[:i], char, chunk_content[i + 1:]
        return chunk_content, '', ''

    def _should_end_sentence(self, sentence: str, mark: str, is_first: bool) -> bool:
        if not sentence or mark not in self.sentence_end_marks: return False
        is_chinese = mark in self.chinese_sentence_end_marks
        if is_first:
            return (len(sentence) > 2 and is_chinese) or (len(sentence.split()) > 1 and not is_chinese)
        return len(sentence) > 4 if is_chinese else (len(sentence.split()) > 4 or (len(sentence.split()) > 2 and mark in {'.', '?', '!'}) )

if __name__ == "__main__":
    node = Node()
    llm_node = LLMNode()

    for event in node:
        status = llm_node(event, node)
        if status == DoraStatus.STOP:
            break