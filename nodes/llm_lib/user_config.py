"""用户配置管理模块"""
import json
from typing import Dict, Optional

from voice_dialogue.utils.logger import logger
from .llm_config import CHINESE_SYSTEM_PROMPT, ENGLISH_SYSTEM_PROMPT
from .paths import USER_PROMPTS_PATH

# 内存缓存，避免重复读取文件
_user_prompts_cache: Optional[Dict[str, str]] = None


def get_user_prompts() -> Dict[str, str]:
    """
    加载用户自定义的 prompts
    
    Returns:
        Dict[str, str]: 用户自定义的 prompts。
    """
    global _user_prompts_cache
    if _user_prompts_cache is not None:
        return _user_prompts_cache

    if not USER_PROMPTS_PATH.exists():
        logger.info(f"用户配置文件不存在，使用空配置: {USER_PROMPTS_PATH}")
        _user_prompts_cache = {}
        return _user_prompts_cache

    try:
        with open(USER_PROMPTS_PATH, 'r', encoding='utf-8') as f:
            user_prompts = json.load(f)
            logger.info("成功从文件加载用户自定义 prompts 到缓存")
            _user_prompts_cache = user_prompts
            return _user_prompts_cache
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"无法加载用户 prompt 配置文件，使用空配置: {e}")
        _user_prompts_cache = {}
        return _user_prompts_cache


def save_user_prompts(prompts: Dict[str, str]) -> bool:
    """
    保存用户自定义的 prompts 到 JSON 文件，并更新缓存。
    
    Args:
        prompts: 要保存的 prompts 字典
        
    Returns:
        bool: 保存是否成功
    """
    global _user_prompts_cache
    try:
        # 确保目录存在
        if not USER_PROMPTS_PATH.parent.exists():
            USER_PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(USER_PROMPTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)
        logger.info(f"用户 prompts 已保存到: {USER_PROMPTS_PATH}")
        _user_prompts_cache = prompts  # 更新缓存
        return True
    except IOError as e:
        logger.error(f"无法保存用户 prompt 配置文件: {e}")
        return False


def get_prompt(language: str) -> str:
    """
    获取指定语言的 prompt，并自动添加 /no_think 指令
    优先从用户配置中获取，如果未配置，则返回默认值
    
    Args:
        language: 语言代码，"zh" 表示中文，其他表示英文
        
    Returns:
        str: 对应语言的系统提示词（已添加 /no_think）
    """
    user_prompts = get_user_prompts()

    if language == "zh":
        base_prompt = user_prompts.get("chinese_prompt", CHINESE_SYSTEM_PROMPT)
    else:
        base_prompt = user_prompts.get("english_prompt", ENGLISH_SYSTEM_PROMPT)

    # 动态添加 /no_think 指令
    # 检查是否已经包含 /no_think，避免重复添加
    if "/no_think" not in base_prompt:
        base_prompt = base_prompt.rstrip() + "\n/no_think"

    return base_prompt


def get_raw_prompt(language: str) -> str:
    """
    获取指定语言的原始 prompt（不添加 /no_think 指令）
    用于API接口返回给前端显示
    
    Args:
        language: 语言代码，"zh" 表示中文，其他表示英文
        
    Returns:
        str: 对应语言的原始系统提示词
    """
    user_prompts = get_user_prompts()

    if language == "zh":
        return user_prompts.get("chinese_prompt", CHINESE_SYSTEM_PROMPT)
    else:
        return user_prompts.get("english_prompt", ENGLISH_SYSTEM_PROMPT)


def reset_prompts_to_default() -> bool:
    """
    重置 prompts 为默认值，并清空缓存。
    
    Returns:
        bool: 重置是否成功
    """
    global _user_prompts_cache
    try:
        if USER_PROMPTS_PATH.exists():
            USER_PROMPTS_PATH.unlink()
            logger.info("用户自定义 prompts 已重置为默认值")
        _user_prompts_cache = {}  # 重置缓存为空字典
        return True
    except IOError as e:
        logger.error(f"重置 prompts 失败: {e}")
        return False
