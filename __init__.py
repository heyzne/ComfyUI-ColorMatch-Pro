"""
ComfyUI ColorMatch Pro - Advanced Color Matching & Cloning for AI Image Generation
自动主体识别 · 颜色匹配追色 · 蒙版区域调色 · 色差修正
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"

# 调试信息
print("[ColorMatch Pro] 节点加载成功:")
for k, v in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {k}: {v}")
