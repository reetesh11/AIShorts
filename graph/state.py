import operator
from typing import Annotated, TypedDict


# ── Sub-structures ────────────────────────────────────────────────────────────

class VoiceoverLine(TypedDict):
    text: str
    tone: str   # excited|mysterious|dramatic|energetic|calm|warm|inspiring


class ScenePlan(TypedDict):
    scene_id: int
    image_description: str          # for image generation
    voiceover_lines: list[VoiceoverLine]
    transition_in: str              # dissolve|cut|fade|zoom_in|zoom_out
    duration_hint: float            # target seconds


class ImagePrompt(TypedDict):
    scene_id: int
    prompt: str
    style_type: str                 # 3d_cartoon|photorealistic|illustration|flat_design|cinematic
    width: int
    height: int


class SceneAudio(TypedDict):
    scene_id: int
    audio_path: str
    word_timings: list[dict]        # [{word, start, end}]
    duration: float


class MetadataOutput(TypedDict):
    title: str                      # best title
    title_options: list[str]        # 3 options
    description: str                # full YouTube description
    hashtags: list[str]
    thumbnail_prompt: str           # image prompt for thumbnail


# ── Main LangGraph state ──────────────────────────────────────────────────────

class ShortsState(TypedDict):
    # ── Input
    topic: str
    cfg: dict

    # ── Research agent
    research_summary: str
    key_facts: list[str]
    target_audience: str
    unique_angle: str

    # ── Script agent
    hook: str
    full_script: str
    cta: str

    # ── Quality gate
    quality_approved: bool
    quality_notes: str

    # ── Screenplay agent (central plan)
    art_style: str
    scene_plans: list[ScenePlan]

    # ── Image prompt agent
    image_prompts: list[ImagePrompt]

    # ── Parallel agents
    generated_images: list[str]     # ordered by scene_id, absolute paths
    scene_audios: list[SceneAudio]  # ordered by scene_id
    metadata: MetadataOutput

    # ── Output agent
    video_path: str
    output_dir: str

    # Accumulates across all parallel agents
    errors: Annotated[list[str], operator.add]
