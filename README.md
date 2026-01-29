# AI Story Video Generator

A fully local AI-powered video generation system that automatically creates story and knowledge-based videos from text keywords. The system operates entirely offline without requiring external APIs, making it suitable for environments with restricted internet access.

## Project Overview

This project automates the complete video production pipeline, transforming a simple keyword into a fully-produced video with:
- AI-generated scripts based on the input topic
- Scene-by-scene image generation using Stable Diffusion
- Text-to-speech audio narration
- Automatic subtitle generation
- Professional video composition and editing

All processing is performed locally using open-source AI models, ensuring privacy and eliminating dependency on cloud services.

## Features

- **Fully Local Operation**: No external API dependencies
- **Automated Script Generation**: Uses Ollama with Qwen 2.5 7B for intelligent story creation
- **Local Image Generation**: Stable Diffusion models (SD 1.5 / SDXL) for scene visualization
- **Text-to-Speech Synthesis**: Coqui TTS or Piper TTS for natural voice narration
- **Automatic Subtitles**: Built-in subtitle generation synchronized with audio
- **Video Composition**: FFmpeg-based automated video editing and assembly
- **One-Command Execution**: Single Python script orchestrates the entire pipeline

## System Requirements

### Required Software

1. **Python 3.8+**
2. **Ollama** - [Download and Install](https://ollama.ai/)
3. **FFmpeg** - [Download and Install](https://ffmpeg.org/download.html)
4. **CUDA** (Optional but highly recommended for GPU acceleration)

### Hardware Recommendations

- **GPU**: NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: At least 20GB free space (for model downloads)

## Installation

### 1. Clone or Download the Project

```bash
cd AIStoryFarm
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA (Critical for GPU Acceleration)

**Important**: If your system has an NVIDIA GPU, you must install the PyTorch CUDA version to enable GPU acceleration. Without CUDA, the system will use CPU mode, which is extremely slow (40+ minutes per image).

#### Check Current PyTorch Version:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If it displays `CUDA available: False`, install the CUDA version:

#### Windows (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Windows (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify Installation:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

This should display `CUDA available: True` and your GPU name.

**Notes**:
- Ensure your NVIDIA drivers are updated to the latest version
- CUDA 12.1 requires NVIDIA driver 525.60.13 or higher
- CUDA 11.8 requires NVIDIA driver 450.80.02 or higher

### 4. Install and Configure Ollama

#### Windows:

1. Download [Ollama Windows version](https://ollama.ai/download/windows)
2. Install Ollama (GUI application)
3. **Ensure Ollama service is running** (check system tray for Ollama icon)
4. Download the model using one of the following methods:

**Method A: Using Command Line (Recommended)**

If the `ollama` command is not available, add Ollama to your system PATH:

a. Locate Ollama installation directory (typically `C:\Users\<username>\AppData\Local\Programs\Ollama`)

b. Add `ollama.exe` directory to system PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Advanced" tab → "Environment Variables"
   - Find `Path` in "System Variables", click "Edit"
   - Click "New", add Ollama installation directory (e.g., `C:\Users\<username>\AppData\Local\Programs\Ollama`)
   - Click "OK" to save
   - **Restart your command prompt** (important!)

c. Verify installation:
   ```bash
   ollama --version
   ```

d. Download model:
   ```bash
   ollama pull qwen2.5:7b
   ```

**Method B: Using Ollama GUI**

1. Open Ollama GUI application
2. Search and download `qwen2.5:7b` model in the interface
3. Wait for download to complete

**Method C: Using Full Path (CMD or PowerShell)**

**In CMD**:
```cmd
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen2.5:7b
```

**In PowerShell**:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b
```

#### Linux/Mac:

```bash
curl https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
```

### 5. Install FFmpeg

#### Windows:

1. Download [FFmpeg Windows version](https://www.gyan.dev/ffmpeg/builds/)
2. Extract and add `bin` directory to system PATH

#### Linux:

```bash
sudo apt-get install ffmpeg
```

#### Mac:

```bash
brew install ffmpeg
```

### 6. Configure TTS (Choose One)

#### Option A: Coqui TTS (Recommended, Auto-Install)

Coqui TTS is automatically installed via `pip install TTS`. The system automatically attempts to use the best available model:
- **XTTS v2** (Priority) - Highest quality, natural voice, multilingual support
- **Tacotron2** (Fallback) - Standard Chinese model
- **FastSpeech2** (Fallback) - Fast generation

Models are automatically downloaded on first run (XTTS v2 ~1.7GB, Tacotron2 ~500MB).

#### Option B: Piper TTS

1. Download [Piper TTS](https://github.com/rhasspy/piper/releases)
2. Download Chinese model:
   ```bash
   # Create model directory
   mkdir -p models/piper/zh_CN
   
   # Download Chinese model (from Piper official)
   # Place model files in models/piper/zh_CN/ directory
   ```

### 7. Stable Diffusion Models (Auto-Download on First Run)

On first run, the program automatically downloads models:
- **SD 1.5 (DreamShaper)**: ~4GB (lightweight, recommended, balanced model)
  - Uses `Lykon/DreamShaper-8`
  - Suitable for diverse subjects: people, animals, objects, scenes
  - Good performance for story illustrations with optimized prompts and style constraints
  - Fallback models: Realistic Vision V5.1 or original SD 1.5
- **SDXL**: ~7GB (high quality, requires more VRAM)

## Usage

### Basic Usage

```bash
python main.py "story keyword"
```

### Interactive Mode

Run without arguments to access an interactive menu:

```bash
python main.py
```

The interactive mode allows you to:
- Select from predefined story topics
- Input custom story text or load from file
- Choose image generation model
- Configure output settings

### Test Image Generation (Custom Prompts)

To test different prompts for image generation:

```bash
# Using Chinese prompt
python test_image_generation.py "一位古代中國老翁坐在傳統木屋內，牆上掛著精美的壁畫"

# Using English prompt (recommended, better model understanding)
python test_image_generation.py "an old Chinese man sitting in a traditional wooden room with beautiful wall paintings, bronze wine cups on the table, sunset light through window"

# Custom parameters
python test_image_generation.py "your prompt" --steps 40 --guidance 10 --style ancient

# View all options
python test_image_generation.py
```

**Prompt Tips**:
- English prompts typically yield better results
- Be specific: include characters, actions, environment, lighting
- Use `--guidance` to adjust strictness (7-12, default 9.0)
- Use `--steps` to adjust quality (20-50, default 30)

### Advanced Options

```bash
# Specify image style
python main.py "historical story" --style chinese_ink

# Specify TTS engine
python main.py "knowledge" --tts piper

# Use SDXL model (requires more VRAM)
python main.py "urban legend" --image-model sdxl

# Custom output filename
python main.py "story keyword" --output my_story
```

### Batch Generation

Generate multiple videos at once:

```bash
# Using predefined list
python batch_generate.py

# Custom keyword list
python batch_generate.py --keywords "idiom story: waiting for rabbit" "history: three visits" "trivia: why is sky blue"

# Specify uniform style
python batch_generate.py --keywords "keyword1" "keyword2" --style cinematic
```

### Parameter Reference

- `keyword`: Topic keyword (required)
- `--style`: Image style
  - `cinematic` (default) - Cinematic style
  - `chinese_ink` - Chinese ink painting
  - `ancient` - Ancient scenes
  - `fantasy` - Fantasy style
  - `horror` - Horror style
  - `hand_drawn` - Hand-drawn style
- `--tts`: TTS engine (`coqui` or `piper`)
- `--image-model`: Image model (`sd15` or `sdxl`)
- `--output`: Output filename (without extension)
- `--lora`: Path to a LoRA weights file or folder (optional; see **FINE_TUNING_GUIDE.md**)
- `--lora-scale`: LoRA strength 0–1 (default 0.8)
- `--checkpoint`: Path to a local full model file, e.g. from CivitAI (see **CIVITAI_IMPORT.md**)

## Technology Stack

### Core Technologies

- **Python 3.8+**: Primary programming language
- **PyTorch**: Deep learning framework for image generation
- **Diffusers**: Hugging Face library for Stable Diffusion models
- **Transformers**: Model loading and inference
- **FFmpeg**: Video processing, encoding, and composition

### AI Models and Services

#### Text Generation (Script Creation)
- **Ollama**: Local LLM server providing REST API interface
- **Qwen 2.5 7B**: Large language model for story generation
  - Generates structured scripts with paragraphs, scenes, and emotions
  - Analyzes story context to recommend visual styles
  - Outputs JSON-formatted script data
  - Model size: ~4.4GB, runs locally via Ollama

#### Image Generation
- **Stable Diffusion 1.5**: Base diffusion model architecture
- **DreamShaper-8**: Fine-tuned model optimized for diverse subjects
- **Realistic Vision V5.1**: Alternative model for realistic scenes
- **SDXL Turbo**: Fast generation variant (1-4 steps)
- **Model Features**:
  - Automatic prompt translation (Chinese to English)
  - Emotional context analysis
  - Style-aware generation
  - Character consistency across scenes
  - Negative prompt optimization to prevent artifacts
  - LoRA support for custom style fine-tuning

#### Text-to-Speech
- **Coqui TTS**: Primary TTS engine
  - XTTS v2: Highest quality, natural voice synthesis with emotional reference audio support
  - Tacotron2: Standard Chinese model
  - FastSpeech2: Fast generation option
- **Piper TTS**: Alternative lightweight TTS engine

#### Video Processing
- **FFmpeg**: Video composition and editing
  - Image-to-video conversion with effects
  - Audio synchronization
  - Subtitle overlay
  - Aspect ratio management (letterboxing)
  - Video concatenation

### Architecture Overview

The system follows a modular pipeline architecture:

1. **Script Generation Module** (`scripts/generate_script.py`)
   - Interfaces with Ollama API
   - Constructs prompts for LLM
   - Parses and validates JSON responses
   - Handles error recovery and JSON repair

2. **Image Generation Module** (`scripts/generate_images.py`)
   - Loads Stable Diffusion models via Hugging Face Diffusers
   - Translates prompts using Google Translator API
   - Manages GPU/CPU device selection
   - Implements prompt engineering with style constraints
   - Supports LoRA and custom checkpoint loading

3. **Audio Generation Module** (`scripts/generate_audio.py`)
   - Manages multiple TTS backends (Coqui, Piper)
   - Handles model fallback chain
   - Supports emotional reference audio for XTTS v2
   - Implements audio normalization and cleanup

4. **Video Generation Module** (`scripts/generate_video.py`)
   - Orchestrates FFmpeg operations
   - Synchronizes audio and video segments
   - Generates subtitle files (SRT format)
   - Applies video effects (zoom, pan, static)
   - Manages aspect ratio and letterboxing

5. **Main Pipeline** (`main.py`)
   - Coordinates all modules
   - Manages file I/O and directory structure
   - Provides interactive and command-line interfaces
   - Handles error propagation and user feedback

### AI Usage Details

#### Script Generation Pipeline

1. **Input Processing**: User provides a keyword or topic
2. **LLM Analysis**: Qwen 2.5 analyzes the topic and generates:
   - Story title
   - Multiple paragraphs with narrative flow
   - Scene descriptions for each paragraph
   - Emotional context analysis
   - Recommended visual style with reasoning
3. **JSON Output**: Structured script data for downstream processing
4. **Validation**: JSON structure validation and error recovery

#### Image Generation Pipeline

1. **Prompt Construction**:
   - Scene description from script
   - Style keywords based on LLM recommendation
   - Emotional vocabulary from context analysis
   - Story title and text as context
   - Character consistency prompts
2. **Translation**: Chinese prompts translated to English for better model understanding
3. **Model Selection**: Automatic fallback chain for model compatibility
4. **Generation**: Stable Diffusion inference with optimized parameters
   - Guidance scale: 7-12 (default 9.0)
   - Inference steps: 20-50 (default 30)
   - Negative prompts prevent artifacts
5. **Quality Control**: Negative prompts prevent artifacts and maintain consistency

#### Audio Generation Pipeline

1. **Text Extraction**: Paragraph text from generated script
2. **TTS Selection**: Automatic model selection (XTTS v2 → Tacotron2 → FastSpeech2)
3. **Synthesis**: Voice generation with natural prosody
   - Optional emotional reference audio for XTTS v2
   - Audio normalization for consistent volume
4. **Duration Calculation**: Audio length used for video synchronization

#### Video Composition Pipeline

1. **Segment Creation**: Each image paired with corresponding audio
2. **Effect Application**: Zoom, pan, or static effects
3. **Synchronization**: Video duration matches audio exactly
4. **Aspect Ratio Management**: Letterboxing to maintain 9:16 format
5. **Subtitle Overlay**: SRT-based subtitles with styling
6. **Concatenation**: All segments combined into final video

## Project Structure

```
AIStoryFarm/
├── main.py                 # Main program entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── batch_generate.py      # Batch processing script
├── test_image_generation.py  # Image generation testing tool
├── scripts/                # Functional modules
│   ├── generate_script.py  # Script generation (LLM)
│   ├── generate_images.py  # Image generation (Stable Diffusion)
│   ├── generate_audio.py   # Audio generation (TTS)
│   └── generate_video.py   # Video generation (FFmpeg)
├── models/                 # Model files (auto-downloaded)
├── data/                   # Data files
│   ├── topics.json        # Predefined story topics
│   └── tts_reference.wav  # Optional emotional reference audio
├── output/                 # Output directory
│   └── {keyword}/
│       ├── script/         # Generated scripts
│       ├── images/         # Generated images
│       ├── audio/          # Generated audio
│       └── video/          # Final videos
├── images/                 # Temporary images (optional)
├── audio/                  # Temporary audio (optional)
└── video/                  # Temporary video (optional)
```

## Troubleshooting

### Issue 1: Ollama Command Not Found (Windows)

**Error**: `'ollama' is not recognized as an internal or external command`

**Solution**:
1. **Confirm Ollama is installed and running**:
   - Check system tray for Ollama icon
   - If not present, launch Ollama from Start menu

2. **Add Ollama to PATH**:
   - Locate Ollama installation: `C:\Users\<your-username>\AppData\Local\Programs\Ollama`
   - Add this directory to system PATH (see installation steps above)
   - **Restart command prompt**

3. **Use full path** (temporary solution):
   
   **In CMD**:
   ```cmd
   "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   ```
   
   **In PowerShell**:
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   ```

4. **Use GUI to download model**:
   - Open Ollama GUI, download model directly in interface

### Issue 2: Ollama Connection Failed

**Error**: `Unable to connect to Ollama`

**Solution**:
1. Confirm Ollama is running (check system tray)
2. Confirm model is downloaded:
   ```bash
   ollama list
   ```
   Or using full path:
   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
   ```
3. Check if Ollama service is running at `http://localhost:11434`
4. If Ollama is not running, launch from Start menu

### Issue 3: FFmpeg Not Found

**Error**: `FFmpeg not available`

**Solution**:
1. Confirm FFmpeg is installed:
   ```bash
   ffmpeg -version
   ```
2. Confirm FFmpeg is in system PATH

### Issue 4: Using CPU Instead of GPU (Extremely Slow Image Generation)

**Symptom**: Image generation shows `device: cpu`, each image takes 40+ minutes

**Solution**:
1. **Check PyTorch CUDA support**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   
2. **If it shows `False`, install PyTorch CUDA version**:
   ```bash
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Or CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
3. **Confirm NVIDIA drivers are installed and updated**
4. **Re-run program**, should display `device: cuda`

### Issue 5: GPU Memory Insufficient

**Error**: `CUDA out of memory`

**Solution**:
1. Use lighter model:
   ```bash
   python main.py "keyword" --image-model sd15
   ```
2. Reduce batch size (modify `generate_images.py`)
3. Close other programs using GPU
4. Use CPU mode (slower, not recommended)

### Issue 6: TTS Generation Failed

**Error**: `Coqui TTS not available` or `Piper TTS not available`

**Solution**:
1. **Coqui TTS**: Confirm installation:
   ```bash
   pip install TTS
   ```
2. **Piper TTS**: Confirm installation and model path configuration

### Issue 7: Generated Images Unrelated to Topic

**Symptom**: Generated images don't match Chinese story content

**Solution**:
1. Check if scene descriptions in script are accurate
2. Try different style options (`--style`)
3. If problem persists, manually edit prompt templates in `scripts/generate_images.py`

### Issue 8: Slow Model Downloads

**Solution**:
1. Use domestic mirrors (if available)
2. Manually download models to `~/.cache/huggingface/` directory
3. Use VPN or proxy

## Customization

### Modify Script Style

Edit the prompt template in `scripts/generate_script.py`. The prompt engineering determines the structure and quality of generated stories.

### Modify Image Style

Edit the `style_prompts` dictionary in `scripts/generate_images.py`. Each style has associated keywords that influence the visual output.

### LoRA Fine-Tuning (Custom Visual Style)

To train a LoRA for a specific visual style (e.g. Chinese ink, anime) and use it in story generation, see **[FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)**. After training, pass the LoRA path with `--lora path/to/lora.safetensors` or set the `LORA_PATH` environment variable.

### Modify Video Effects

Edit effect parameters in `scripts/generate_video.py`. Effects include zoom, pan, and static positioning for each video segment.

## Output Specifications

Generated videos are saved at:
```
output/{keyword}/video/{keyword}_with_subtitles.mp4
```

Video Specifications:
- Resolution: 1080x1920 (Shorts format)
- Frame Rate: 30 FPS
- Format: MP4 (H.264 + AAC)
- Aspect Ratio: 9:16 (vertical video)

## Workflow

1. **Input Keyword** → User provides topic
2. **Generate Script** → Ollama + Qwen generates story paragraphs
3. **Generate Images** → Stable Diffusion generates background images for each scene
4. **Generate Audio** → TTS synthesizes voice narration
5. **Compose Video** → FFmpeg combines all elements
6. **Output Video** → Final MP4 file

## Usage Recommendations

1. **First Run**: Recommend using `--image-model sd15` (lighter)
2. **GPU Acceleration**: Ensure CUDA is correctly installed for optimal performance
3. **Batch Generation**: Can write scripts to loop `main.py` for batch processing
4. **Customization**: Modify module parameters and prompts as needed

## Technical Implementation Notes

### Model Loading Strategy

The system implements a fallback chain for model loading:
- Primary: DreamShaper-8 (if available locally)
- Secondary: Realistic Vision V5.1
- Tertiary: Original Stable Diffusion 1.5

This ensures compatibility across different hardware configurations.

### Memory Management

- CUDA memory allocation uses expandable segments to reduce fragmentation
- Models are loaded once and reused across multiple image generations
- Automatic device selection (CUDA > CPU) with fallback

### Error Handling

- JSON parsing includes repair mechanisms for incomplete LLM responses
- Model loading includes fallback chains
- TTS includes automatic model selection based on availability
- All modules include comprehensive error messages for debugging

## License

This project is for educational and research purposes only.

## Contributing

Issues and Pull Requests are welcome!

## Contact

For issues, please submit an Issue on GitHub.

---

**Note**: This system runs entirely locally and does not depend on any external APIs, making it suitable for environments with restricted internet access.
