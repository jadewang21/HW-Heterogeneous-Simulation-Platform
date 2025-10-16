"""
Flash Attention配置模块
用于控制是否使用修正后的Flash Attention实现
"""

# 全局配置：是否使用修正后的Flash Attention实现
USE_CORRECTED_FLASH_ATTENTION = True

def get_flash_attention_implementation():
    """
    返回当前使用的Flash Attention实现
    """
    if USE_CORRECTED_FLASH_ATTENTION:
        from GenZ.Models.attention import (
            mha_flash_attention_corrected_prefill,
            mha_flash_attention_corrected_decode
        )
        return mha_flash_attention_corrected_prefill, mha_flash_attention_corrected_decode
    else:
        from GenZ.Models.attention import (
            mha_flash_attention_prefill,
            mha_flash_attention_decode
        )
        return mha_flash_attention_prefill, mha_flash_attention_decode

def set_flash_attention_mode(use_corrected=True):
    """
    设置Flash Attention实现模式
    
    Args:
        use_corrected (bool): 是否使用修正后的实现
    """
    global USE_CORRECTED_FLASH_ATTENTION
    USE_CORRECTED_FLASH_ATTENTION = use_corrected

def is_corrected_flash_attention_enabled():
    """
    检查是否启用了修正后的Flash Attention实现
    """
    return USE_CORRECTED_FLASH_ATTENTION
