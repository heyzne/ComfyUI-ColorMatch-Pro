# ComfyUI-ColorMatch-Pro-Enhanced

融合 ColorMatch Pro + mini-nodes 算法的高级颜色匹配插件

## 核心改进

### 1. 智能颜色差异自适应
- 自动检测原图与参考图的颜色差异（LAB空间AB通道）
- **差异 < 3**: 几乎不改（保留原图）
- **差异 3~8**: 轻微校正
- **差异 8~15**: 适度校正
- **差异 > 15**: 全力校正

### 2. 融合 mini-nodes 算法
新增 4 种来自 mini-nodes 的调色模式：

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **mini_linear** | RGB独立缩放，对比度保留最好 | 遮罩校色 |
| **mini_mean** | 仅平移均值，保留原图对比度 | 风格参考稳定 |
| **mini_mkl** | 协方差矩阵映射，通用全局 | 快捷校色 |
| **mini_wavelet** | 低频校色，还原最自然 | 相同构图矫正偏色 |

### 3. 新增 SmartColorClone 节点
一键智能调色，自动：
1. 检测颜色差异
2. 选择最佳调色模式
3. 自动排除白色背景干扰
4. 输出调试信息（差异值、使用模式）

### 4. 产品专用模式
- 自动排除参考图白色/近白背景
- 只提取产品本身颜色进行匹配
- 避免背景色干扰调色结果

## 节点列表

| 节点 | 功能 |
|------|------|
| 🎨 Color Match | 基础追色匹配（8种算法） |
| 🔍 Auto Subject Mask | 自动主体检测蒙版 |
| 🎭 Color Clone | 高级颜色克隆（13种模式） |
| 🧠 Smart Color Clone | 一键智能追色（推荐） |
| 🎯 Regional Color Match | 区域调色 |
| 🔧 Mask Tool | 蒙版处理工具 |
| ⚡ Color Correction | 色差修正 |
| 📦 Batch Color Match | 批量追色 |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/heyzne/ComfyUI-ColorMatch-Pro.git ComfyUI-ColorMatch-Pro-Enhanced
```

或使用本压缩包直接解压到 `custom_nodes` 目录。

## 依赖

```bash
pip install scikit-learn scipy opencv-python pillow numpy
```

## 使用建议

### 产品换装场景（您的需求）
```
模特图 + 参考产品图（白底）
    → SegmentAnything 分割出内衣区域蒙版
    → Smart Color Clone（自动检测差异并调色）
    → 如果颜色本来对了，几乎不改；颜色不对则自动追色
```

### 推荐模式选择
- **颜色本来对了，不想改**: 用 `SmartColorClone` + auto_skip=True
- **轻微偏色**: `mini_mean` 或 `color_adapter`
- **明显偏色**: `product` 或 `mini_mkl`
- **光影不一致**: `mini_wavelet`
