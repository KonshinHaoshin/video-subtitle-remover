# Video Subtitle Remover — 现代化 UI & 使用说明

一款基于 **PyQt5 + OpenCV** 的视频字幕去除器，提供**现代深色主题**与**所见即所得的矩形框选**工作流。支持单/多文件处理，实时预览与进度反馈。

## ✨ 特性概览
- 🖱️ **拖拽框选**：在预览上按下并拖拽即可圈定字幕区域，自动映射到原视频分辨率；右侧滑块支持像素级微调  
- 📹 **视频/图片**：支持常见视频格式与图片，首帧预览，时间轴拖动定位  
- 🧠 **智能复用**：批量处理时若分辨率一致，自动复用同一字幕区域  
- ⚙️ **无阻塞**：延迟导入 backend（如需 torch）并在后台线程处理，不阻塞 UI  
- 🎛️ **现代 UI**：深色主题、扁平化控件、自定义 QSS

## 🧱 技术栈
- Python 3.9+（建议 3.10/3.11）
- PyQt5（界面与交互）
- OpenCV（预览/帧处理）

> 注：示例 `backend/main.py` 提供了一个**无 torch 的演示实现**（对选区做 inpaint/模糊并导出 `_nosub` 文件）。你可以替换为自己的算法实现。

## 📦 安装与环境
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## 🚀 运行
```bash
python gui_pyqt.py
```

流程：
1) 点击工具栏 **打开** 选择视频/图片  
2) 在预览区**按下并拖拽**框选字幕区域（松开即生效）  
3) 右侧滑块微调（X/Y 起点与宽/高）  
4) 点击 **Run**，观察右侧日志与进度条  

## 🖼 映射与等比缩放
预览采用 **letterbox**（等比缩放+黑边）。框选坐标会正确映射回原始像素，无惧窗口尺寸变化。

## 🧩 目录结构
```
VideoSubtitleRemover/
├─ gui.py # 源ui文件
├─ gui_pyqt.py # 新增
├─ style.qss # 新增
├─ requirements.txt
├─ backend/
│  ├─ main.py
│  └─ tools/common_tools.py

```

## 🗺 Roadmap
- [ ] 选区可拖拽移动/四角拉伸进行二次编辑
- [ ] 主题切换（深/浅）与自定义配色
- [ ] 快捷键：方向键微调、Ctrl+S 保存选区
- [ ] 批处理任务队列与失败重试
- [ ] 算法模块热插拔

