# AI Shorts Agent

Generate complete 60-second YouTube Shorts from a single topic string using a 10-agent LangGraph pipeline.

**Free by default** — Groq LLMs (free tier) + Pollinations.ai images (no key needed) + edge-tts.

## What it produces

- AI-researched, SEO-optimised script with viral hook
- 5–6 AI-generated 9:16 portrait images (FLUX via Pollinations, or Imagen 3)
- Voiceover with per-line tone prosody (excited, dramatic, calm, etc.)
- Final MP4 with karaoke-style captions (active-word highlighted yellow)
- YouTube metadata: 3 title options, full description, hashtags, thumbnail prompt

## Pipeline

```
research → script → quality_gate → screenplay → image_prompt
         → image_gen → voiceover → metadata → stitch → output
```

| Agent | Role | Default model |
|---|---|---|
| `research` | Topic facts, viral angle, audience | Groq llama-3.3-70b |
| `script` | 120–150 word script with hook + CTA | Groq llama-3.3-70b |
| `quality_gate` | Scores hook/pacing/CTA, improves weak hooks | Groq llama-3.1-8b-instant |
| `screenplay` | 5–6 scene plans with transitions and timing | Groq llama-3.3-70b |
| `image_prompt` | Detailed FLUX/Imagen prompts per scene | Groq llama-3.3-70b |
| `image_gen` | AI image generation (parallel, 3 concurrent) | FLUX / Imagen 3 |
| `voiceover` | Edge TTS with tone-modulated prosody | edge-tts Aria Neural |
| `metadata` | Titles, description, hashtags, thumbnail | Groq llama-3.1-8b-instant |
| `stitch` | Assembles MP4 with captions and transitions | moviepy |
| `output` | Saves `description.md` + `meta.json` | — |

## Setup

```bash
git clone https://github.com/reetesh11/AIShorts.git
cd AIShorts

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your GROQ_API_KEY
```

## API Keys

| Key | Required | Free tier | Get it at |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (default config) | 14,400 req/day | [console.groq.com](https://console.groq.com) |
| `GOOGLE_API_KEY` | Only for Imagen 3 images | No | [console.cloud.google.com](https://console.cloud.google.com) |
| `ANTHROPIC_API_KEY` | Only if `provider: anthropic` | No | [console.anthropic.com](https://console.anthropic.com) |

## Usage

```bash
# Full video generation
python main.py "The speed of light explained simply"

# Dry run — text agents only (no images, no video)
python main.py --dry-run "Quantum computing basics"

# Custom config file
python main.py --config config/config.yaml "Black holes for beginners"

# Save full pipeline state to JSON for inspection
python main.py --output-json state.json "Mars colonization"
```

## Configuration

`config/config.yaml` controls everything:

```yaml
research:
  max_words: 120          # Keep tight for 60s shorts

agents:
  script:
    provider: groq
    model: llama-3.3-70b-versatile
    temperature: 0.7

image_generation:
  provider: pollinations   # Free, no key — or switch to "google" for Imagen 3

tts:
  voice: "en-US-AriaNeural"

video:
  width: 1080
  height: 1920
  fps: 30
  caption_font_size: 58
```

### Switching to Imagen 3 (better image quality)

```yaml
image_generation:
  provider: google
  model: imagen-3.0-generate-002
  aspect_ratio: "9:16"
```

Requires `GOOGLE_API_KEY` with billing enabled.

## Output structure

```
outputs/{topic_slug}/
├── short.mp4            # Final 9:16 YouTube Short
├── description.md       # Script, screenplay, metadata — ready to copy-paste
├── meta.json            # Title, hashtags, thumbnail prompt (compact JSON)
├── audio/
│   └── scene_01.mp3     # Per-scene TTS audio
└── images/
    └── scene_01.png     # AI-generated scene image
```

## Requirements

- Python 3.11+
- ffmpeg (bundled automatically via `imageio-ffmpeg`)
- macOS or Linux (Windows untested)
