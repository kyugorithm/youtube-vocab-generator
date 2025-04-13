
# 📘 YouTube Advanced Vocabulary Generator

This tool creates vocabulary lists from YouTube videos by downloading the audio, transcribing it with Whisper, and using Claude AI to extract advanced expressions and create vocabulary cards.

---

## ✨ Features

- **Automatic Audio Extraction**: Downloads audio from YouTube videos
- **Speech-to-Text**: Transcribes audio using WhisperX
- **Advanced Vocabulary**: Uses Claude AI to identify and explain advanced expressions
- **Sequential Numbering**: Automatically numbers expressions for easy reference
- **Multi-language Support**: Create vocabulary cards for various target languages
- **Caching System**: Processes only necessary steps, skipping those already completed
- **Markdown Output**: Produces clean, formatted vocabulary cards

---

## 📁 Directory Structure

```
ENHANCE_ENGLISH/
├── output/
│   └── [VIDEO_ID]/           # e.g. Lhpu3GdlV3w
│       ├── 1_audio/
│       │   └── [video_title].wav
│       ├── 2_transcript/
│       │   └── transcription.txt
│       └── 3_vocabulary/
│           ├── raw_response.txt             # Original Claude response
│           └── YYYYMMDD_VideoTitle.md       # Formatted vocabulary cards with date
├── resources/
│   ├── claude_api_key.txt     # Your Claude API key (keep private)
│   └── prompt.txt             # Prompt template for Claude
├── main.py                    # Main script
└── readme.md                  # Documentation
```

---

## 🛠️ Prerequisites

- Python 3.8+
- FFmpeg
- Claude API key

---

## 📦 Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/enhance-english.git
cd enhance-english
```

### 2. Create a conda environment (recommended)

```bash
conda create -n vocab-gen python=3.10
conda activate vocab-gen
```

### 3. Install dependencies

```bash
conda install -c conda-forge ffmpeg
conda install -c pytorch pytorch torchaudio
pip install yt-dlp anthropic
pip install git+https://github.com/m-bain/whisperx.git
```

### 4. Set up resources

- Create a `resources` directory
- Save your Claude API key in `resources/claude_api_key.txt`
- Ensure `resources/prompt.txt` exists with the proper prompt format

---

## ▶️ Usage

### Basic Usage

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### All Options

```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output_dir "output" \
  --whisper_model "turbo" \
  --target_language "Korean" \
  --force
```

### Parameters

- `--url`: YouTube video URL (**required**)
- `--output_dir`: Output directory (default: `"output"`)
- `--whisper_model`: Whisper model size (default: `"turbo"`)
- `--target_language`: Target language for vocabulary (default: `"Korean"`)
- `--force`: Force reprocessing of already processed steps

---

## 🧠 Available Whisper Models

| Model      | Description                            |
|------------|----------------------------------------|
| `tiny`     | Very fast, less accurate (~75MB)       |
| `base`     | Fast, decent accuracy (~142MB)         |
| `small`    | Balanced speed/accuracy (~466MB)       |
| `medium`   | Good accuracy, slower (~1.5GB)         |
| `large-v2` | Best accuracy, slowest (~3GB)          |
| `turbo`    | Fast + accurate balance (~1.62GB) ✅ Recommended |

---

## 🛠️ Technical Notes

- The script uses **CPU by default** for maximum compatibility.
- On **Mac**, MPS (Metal Performance Shaders) acceleration is not officially supported by WhisperX. For faster performance, try Whisper via Hugging Face with MPS.
- Claude API official docs: https://docs.anthropic.com/claude/docs/models-overview
- Whisper turbo model size: ~1.62GB
- Recommended Claude model: `claude-3-7-sonnet-latest`

---

## ⚡ Performance Acceleration

⚠️ By default, this script runs on **CPU**, which may lead to **slow processing**, especially with long videos or large Whisper models.  
If your system supports GPU acceleration or Apple Silicon, we recommend enabling hardware acceleration for significantly faster transcription.

### Supported Acceleration Options

| Platform           | Acceleration Method           | Notes |
|--------------------|-------------------------------|-------|
| 🖥️ NVIDIA GPU       | `device="cuda"`                | Requires CUDA-compatible PyTorch and GPU |
| 🍎 Apple Silicon    | `device="mps"`                 | Uses Metal Performance Shaders (MPS) backend |
| 🧠 CPU Only         | Default: `device="cpu"`        | Universally supported but slowest |

### Auto-Select Device Example

```python
import torch

device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

model = whisperx.load_model("large-v2", device)
```

> 💡 Tip: Using GPU or Apple Silicon acceleration can reduce transcription time from several minutes to seconds, depending on the video length and model size.

---

## ⚖️ Legal Disclaimer

**Important**: Downloading content from YouTube without permission may violate YouTube's Terms of Service and potentially copyright law. This tool should only be used with:

- Content you own or have created
- Content explicitly licensed under Creative Commons
- Content in the public domain
- Content for which you have received explicit permission from the copyright holder
- Content that falls under fair use in your jurisdiction (e.g., educational use)

YouTube's Terms of Service specifically prohibit:

> "access, reproduce, download, distribute, transmit, broadcast, display, sell, license, alter, modify or otherwise use any part of the Service or any Content except: (a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders."

The developers of this tool are not responsible for any misuse or legal consequences that may arise from using this software.  
Users are solely responsible for ensuring their use complies with all applicable laws and terms of service.

---

## 🔐 Security Notice

**Never share your Claude API key.**  
If your key is exposed, revoke it immediately and generate a new one from your [Anthropic dashboard](https://console.anthropic.com/).

---

## 📝 License

This project is licensed under the **[MIT License](LICENSE)**.



<img width="958" alt="image" src="https://github.com/user-attachments/assets/30b1ec06-69c9-4c05-a199-6fb9dea0b6b5" />
