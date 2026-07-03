"""
ComfyUI ColorMatch Pro Enhanced - 融合 mini-nodes 算法的高级颜色匹配插件
自动主体识别 · 颜色匹配追色 · 蒙版区域调色 · 色差修正 · 智能自适应

融合算法来源：
- ColorMatch Pro (heyzne): LAB直方图匹配、K-Means、KNN等
- mini-nodes (catmaxzj): Linear/Mean/MKL/Wavelet调色算法

核心改进：
1. 颜色差异自适应：颜色接近时自动降低调色强度/跳过
2. 产品专用模式：自动排除白色背景干扰
3. 新增 SmartColorClone 一键智能调色节点
4. 新增 mini-nodes 的 4 种调色算法
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"

# 调试信息
print("[ColorMatch Pro Enhanced] 节点加载成功:")
for k, v in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f" - {k}: {v}")
