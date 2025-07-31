"""LLM模型配置管理"""

from typing import Dict, Any

from voice_dialogue.utils.apple_silicon import get_optimal_llama_cpp_config, get_apple_silicon_info

__all__ = ('get_llm_model_params', 'get_apple_silicon_summary', 'CHINESE_SYSTEM_PROMPT', 'ENGLISH_SYSTEM_PROMPT')

CHINESE_SYSTEM_PROMPT = (
    "你是AI助手。请以自然流畅的中文口语化表达直接回答问题，避免冗余的思考过程。"
    "你的回答第一句话必须少于十个字。每段回答控制在二到三句话，既不要过短也不要过长，以适应对话语境。"
    "回答应准确、精炼且有依据。"
)

ENGLISH_SYSTEM_PROMPT = (
    "You are an AI assistant. "
    "Please answer directly and naturally, using conversational English, without showing your thinking process. "
    "Your first sentence must be less than 10 words. "
    "Your responses should be accurate, concise, and well-supported, ideally around 2-3 sentences long to ensure a good conversational flow."
)


def get_llm_model_params() -> Dict[str, Any]:
    """
    获取LLM模型参数，基于Apple Silicon芯片信息动态配置
    
    Returns:
        Dict[str, Any]: LLM模型参数配置
    """
    # 获取Apple Silicon优化配置
    optimal_config = get_optimal_llama_cpp_config()

    # 基础模型参数
    model_params = {
        'streaming': True,
        'n_gpu_layers': -1,
        'n_batch': 1024,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'max_tokens': 32768,
        'model_kwargs': {
            'mini_p': 0,
            'presence_penalty': 1.5
        },
        'verbose': False,
    }

    # 应用Apple Silicon优化配置
    model_params.update(optimal_config)

    return model_params


def get_apple_silicon_summary() -> Dict[str, Any]:
    """
    获取Apple Silicon芯片信息摘要
    
    Returns:
        Dict[str, Any]: 芯片信息摘要
    """
    chip_info = get_apple_silicon_info()
    optimal_config = get_optimal_llama_cpp_config()

    return {
        'chip_name': chip_info.chip_name,
        'is_apple_silicon': chip_info.is_apple_silicon,
        'total_cores': chip_info.total_cores,
        'performance_cores': chip_info.performance_cores,
        'efficiency_cores': chip_info.efficiency_cores,
        'memory_gb': chip_info.memory_gb,
        'gpu_cores': chip_info.gpu_cores,
        'optimal_n_threads': optimal_config['n_threads'],
        'optimal_n_ctx': optimal_config['n_ctx'],
        'config_note': '仅使用性能核心(P-cores)以获得最佳性能' if chip_info.is_apple_silicon else '标准配置'
    }


# 预设配置模板
LLAMA_CPP_CONFIG_PRESETS = {
    'high_performance': {
        'description': '高性能配置 - 适合Apple M1 Pro/Max及以上',
        'n_ctx': 8192,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 20,
    },
    'balanced': {
        'description': '平衡配置 - 适合Apple M1/M2基础版',
        'n_ctx': 4096,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 20,
    },
    'memory_efficient': {
        'description': '内存优化配置 - 适合内存较小的设备',
        'n_ctx': 2048,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 20,
    }
}


def get_config_preset(preset_name: str) -> Dict[str, Any]:
    """
    获取预设配置
    
    Args:
        preset_name: 预设名称 ('high_performance', 'balanced', 'memory_efficient')
    
    Returns:
        Dict[str, Any]: 预设配置
    """
    if preset_name not in LLAMA_CPP_CONFIG_PRESETS:
        raise ValueError(f"未知的预设配置: {preset_name}")

    preset = LLAMA_CPP_CONFIG_PRESETS[preset_name].copy()

    # 移除描述字段
    preset.pop('description', None)

    # 添加n_threads配置（基于当前硬件）
    optimal_config = get_optimal_llama_cpp_config()
    preset['n_threads'] = optimal_config['n_threads']

    return preset
