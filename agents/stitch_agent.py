"""
Stitch Agent — assembles the final 9:16 MP4.
Uses screenplay's per-scene transition specs and per-scene audio for precise sync.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from graph.state import ShortsState

_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_PATHS:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[size] = f
            return f
        except Exception:
            pass
    return ImageFont.load_default()


def _build_chunks(word_timings: list[dict], chunk_size: int = 4) -> list[dict]:
    chunks = []
    for i in range(0, len(word_timings), chunk_size):
        group = word_timings[i: i + chunk_size]
        chunks.append({
            "words":       [w["word"] for w in group],
            "word_starts": [w["start"] for w in group],
            "start":       group[0]["start"],
            "end":         group[-1]["end"],
        })
    return chunks


def _chunk_at(t: float, chunks: list[dict]) -> tuple[list[str], int]:
    for chunk in chunks:
        if chunk["start"] <= t <= chunk["end"] + 0.45:
            active = 0
            for j, ws in enumerate(chunk["word_starts"]):
                if ws <= t:
                    active = j
            return chunk["words"], active
    return [], -1


def _render_caption(frame: np.ndarray, words: list[str], active_idx: int,
                    font_size: int, width: int, height: int) -> np.ndarray:
    if not words:
        return frame
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    space_w    = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = [draw.textbbox((0, 0), w, font=font)[2] for w in words]
    total_w    = sum(word_widths) + space_w * (len(words) - 1)
    max_w      = width - 80

    if total_w <= max_w:
        lines = [list(range(len(words)))]
    else:
        mid   = (len(words) + 1) // 2
        lines = [list(range(mid)), list(range(mid, len(words)))]

    line_h   = int(font_size * 1.35)
    base_y   = int(height * 0.76) - (len(lines) * line_h) // 2

    for li, indices in enumerate(lines):
        line_w = sum(word_widths[i] for i in indices) + space_w * (len(indices) - 1)
        x = (width - line_w) // 2
        y = base_y + li * line_h
        for j, wi in enumerate(indices):
            color = (255, 230, 0) if wi == active_idx else (255, 255, 255)
            draw.text((x, y), words[wi], font=font, fill=color,
                      stroke_width=3, stroke_fill="black")
            x += word_widths[wi] + (space_w if j < len(indices) - 1 else 0)

    return np.array(img)


def _resize_to_portrait(img_path: str, width: int, height: int) -> str:
    img = Image.open(img_path).convert("RGB")
    ir, tr = img.width / img.height, width / height
    if ir > tr:
        nw, nh = int(img.width * height / img.height), height
    else:
        nw, nh = width, int(img.height * width / img.width)
    img = img.resize((nw, nh), Image.LANCZOS)
    l = (img.width - width) // 2
    t = (img.height - height) // 2
    img = img.crop((l, t, l + width, t + height))
    out = str(Path(img_path).parent / f"r_{Path(img_path).name}")
    img.save(out)
    return out


def _make_caption_processor(global_start: float, chunks: list[dict],
                             font_size: int, w: int, h: int):
    def process(get_frame, t):
        frame = get_frame(t)
        words, active = _chunk_at(global_start + t, chunks)
        return _render_caption(frame, words, active, font_size, w, h)
    return process


def stitch_node(state: ShortsState) -> dict:
    from moviepy.editor import (
        ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip,
    )

    cfg      = state["cfg"]
    vid_cfg  = cfg["video"]
    width, height = vid_cfg["width"], vid_cfg["height"]
    fps      = vid_cfg.get("fps", 30)
    fade_dur = vid_cfg.get("fade_duration", 0.5)
    font_size = vid_cfg.get("caption_font_size", 65)

    images      = state.get("generated_images", [])
    scene_audios = state.get("scene_audios", [])
    scene_plans  = state.get("scene_plans", [])
    topic_slug   = state["topic"][:30].lower().replace(" ", "_").replace("/", "_")

    output_dir = Path(cfg["video"]["output_dir"]) / topic_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "short.mp4")

    if not images:
        return {"errors": ["stitch_agent: no generated images"], "video_path": ""}
    if not scene_audios:
        return {"errors": ["stitch_agent: no scene audios"], "video_path": ""}

    # Build scene_id → transition map
    transition_map = {s["scene_id"]: s.get("transition_in", "dissolve") for s in scene_plans}

    # Align images and audios by index (both ordered by scene_id)
    pairs = list(zip(images, scene_audios))
    if not pairs:
        return {"errors": ["stitch_agent: image/audio count mismatch"], "video_path": ""}

    try:
        print("  Stitching video...")
        resized = [_resize_to_portrait(p, width, height) for p, _ in pairs]

        # Build per-scene caption chunks using global timeline
        # Compute global start times for each scene
        global_starts = []
        t = 0.0
        for _, sa in pairs:
            global_starts.append(t)
            t += sa["duration"]

        clips = []
        for i, ((_, sa), img_path) in enumerate(zip(pairs, resized)):
            scene_id   = sa["scene_id"]
            duration   = sa["duration"]
            transition = transition_map.get(scene_id, "dissolve")
            g_start    = global_starts[i]

            # Build word-timing chunks for this scene (timings are already global)
            chunks = _build_chunks(sa["word_timings"], chunk_size=4)

            # Base image clip
            clip = (
                ImageClip(img_path)
                .set_duration(duration)
                .fl(_make_caption_processor(g_start, chunks, font_size, width, height))
            )

            # Apply transition
            if i > 0:
                if transition in ("dissolve", "fade"):
                    clip = clip.crossfadein(fade_dur)
                # cut = no modification (instant switch via padding=0)

            clip = clip.set_audio(AudioFileClip(sa["audio_path"]))
            clips.append((clip, transition))

        # Concatenate with per-transition padding
        final_clips = [clips[0][0]]
        padding_sum = 0.0
        for clip, transition in clips[1:]:
            if transition in ("dissolve", "fade"):
                final_clips.append(clip)
                padding_sum = -fade_dur
            else:
                final_clips.append(clip)
                padding_sum = 0.0

        video = concatenate_videoclips(
            final_clips,
            padding=-fade_dur,
            method="compose",
        )

        video.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(output_dir / "temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
        print(f"  Video saved: {output_path}")
        return {"video_path": output_path}

    except Exception as e:
        return {"errors": [f"stitch_agent: {e}"], "video_path": ""}
