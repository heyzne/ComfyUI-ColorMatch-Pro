# 🎨 ComfyUI ColorMatch Pro

高级颜色匹配与克隆系统，专为AI图像生成优化。解决ComfyUI出图色差问题，实现自动主体识别、智能追色、蒙版区域调色。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| **🎨 智能追色** | 6种颜色匹配算法（自适应/直方图/LAB/Reinhard/MKL/IDT） |
| **🔍 主体识别** | 自动检测主体并生成蒙版（显著性/人脸/色彩聚类/边缘） |
| **🎭 颜色克隆** | 模仿参考图颜色风格（全图/色调/氛围/调色板） |
| **🎯 区域调色** | 支持蒙版指定区域独立调色，背景可单独处理 |
| **⚡ 色差修正** | 自动白平衡、色调校正、对比度调整 |
| **🔧 蒙版工具** | 羽化/扩展/收缩/平滑/反转/阈值/混合操作 |
| **📦 批量处理** | 支持多图批量颜色匹配 |

## 🚀 快速安装

```bash
# 进入ComfyUI自定义节点目录
cd ComfyUI/custom_nodes/

# 克隆仓库
git clone https://github.com/yourusername/ComfyUI-ColorMatch-Pro.git

# 安装依赖
cd ComfyUI-ColorMatch-Pro
pip install -r requirements.txt

# 重启ComfyUI
```

## 📖 使用说明

### 基础追色工作流
1. 加载 `Color Match (追色匹配)` 节点
2. 连接 `image`（待调整图）和 `reference`（参考图）
3. 选择匹配算法，推荐 `adaptive`（自适应）
4. 调整 `strength` 控制效果强度

### 自动主体识别
1. 使用 `Auto Subject Mask (自动主体蒙版)` 节点
2. 选择检测方法：`auto`（智能）/`saliency`（显著性）/`face`（人脸）
3. 输出蒙版可直接接入追色节点

### 区域调色
1. 使用 `Regional Color Match (区域调色)` 节点
2. 输入 `region_mask` 指定调色区域
3. 可选输入 `background_reference` 单独调整背景

### 颜色克隆模式
- **full**: 完整颜色风格克隆
- **tone**: 仅克隆色调，保留结构
- **mood**: 克隆光影氛围
- **palette**: 基于调色板的智能匹配

## 🧪 算法说明

### 颜色匹配算法
- **Adaptive**: 智能分析图像特征，自动选择最佳算法
- **Histogram**: 直方图匹配，适合大色调差异
- **LAB Transfer**: LAB空间统计迁移，自然柔和
- **Reinhard**: 对数空间迁移，保留细节
- **MKL**: Monge-Kantorovitch线性变换，精确匹配
- **IDT**: 迭代分布迁移，渐进式匹配

### 主体检测方法
- **Auto**: 组合显著性和边缘检测，智能选择最大主体
- **Saliency**: 视觉显著性检测
- **Face**: 人脸检测（优先），无人脸时回退显著性
- **Color Cluster**: K-Means色彩聚类，分离主体背景
- **Edge**: 基于边缘轮廓检测

## 🎛️ 节点列表

| 节点 | 输入 | 输出 | 用途 |
|------|------|------|------|
| Color Match | image, reference, method, strength | image, mask | 基础追色 |
| Auto Subject Mask | image, method, expand | mask, preview | 自动主体检测 |
| Color Clone | image, reference, mode, strength | image | 风格克隆 |
| Regional Color Match | image, mask, reference, method | image, mask | 区域调色 |
| Mask Tool | mask, operation, amount | mask | 蒙版处理 |
| Color Correction | image, target_color, strength | image | 色差修正 |
| Batch Color Match | images, references, method | images | 批量处理 |

## 📁 工作流示例

`workflows/` 目录包含预设工作流：
- `basic_color_match.json` - 基础追色
- `mask_regional_toning.json` - 蒙版区域调色
- `advanced_color_clone.json` - 高级颜色克隆

## 🖼️ 效果展示

![Demo](assets/demo.png)

## 🤝 贡献

欢迎提交Issue和PR！请确保代码通过基本测试。

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)
