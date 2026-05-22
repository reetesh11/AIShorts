"""
Voiceover Agent — generates per-scene audio files with tone-based prosody.
Each scene's voiceover lines are spoken with their individual tone.
Returns scene_audios: list of {scene_id, audio_path, word_timings, duration}.
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from graph.state import ShortsState

TONE_PROSODY: dict[str, tuple[str, str]] = {
    "excited":    ("+28%", "+10Hz"),
    "mysterious": ("-22%", "-5Hz"),
    "dramatic":   ("-12%", "-3Hz"),
    "energetic":  ("+22%", "+6Hz"),
    "calm":       ("-8%",  "+0Hz"),
    "warm":       ("+5%",  "+2Hz"),
    "inspiring":  ("+12%", "+4Hz"),
    "neutral":    ("+10%", "+0Hz"),
}


def _words_from_sentence_boundaries(
    boundaries: list[dict],
    time_offset: float = 0.0,
) -> list[dict]:
    result = []
    for sb in boundaries:
        sent_start = sb["offset"] / 10_000_000 + time_offset
        sent_dur   = sb["duration"] / 10_000_000
        raw_words  = sb["text"].split()
        if not raw_words:
            continue
        weights = []
        for w in raw_words:
            base = max(len(w.strip(".,!?;:'\"")) , 2)
            if w.endswith((".", "!", "?")):
                base += 4
            elif w.endswith(","):
                base += 2
            weights.append(base)
        total_w = sum(weights)
        t = sent_start
        for w, wt in zip(raw_words, weights):
            slot = (wt / total_w) * sent_dur
            result.append({
                "word":  w.strip(".,!?;:'\"") or w,
                "start": round(t, 4),
                "end":   round(t + slot * 0.78, 4),
            })
            t += slot
    return result


async def _generate_line_audio(
    text: str, voice: str, rate: str, pitch: str, volume: str, output_path: str
) -> list[dict]:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    audio_data = b""
    boundaries: list[dict] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
        elif chunk["type"] == "SentenceBoundary":
            boundaries.append(chunk)
    Path(output_path).write_bytes(audio_data)
    return _words_from_sentence_boundaries(boundaries)


async def _generate_scene_audio(
    scene: dict, voice: str, volume: str, output_path: str
) -> tuple[list[dict], float]:
    """Generate audio for all voiceover lines in a scene, concatenate, return (word_timings, duration)."""
    lines = scene.get("voiceover_lines", [])
    if not lines:
        return [], 0.0

    tmp_dir = tempfile.mkdtemp()
    line_files: list[str] = []
    all_timings: list[dict] = []
    current_offset = 0.0

    for i, line in enumerate(lines):
        tone       = line.get("tone", "neutral")
        rate, pitch = TONE_PROSODY.get(tone, ("+10%", "+0Hz"))
        tmp_path   = os.path.join(tmp_dir, f"line_{i:02d}.mp3")

        timings = await _generate_line_audio(
            text=line["text"], voice=voice, rate=rate,
            pitch=pitch, volume=volume, output_path=tmp_path,
        )

        for wt in timings:
            all_timings.append({
                "word":  wt["word"],
                "start": round(current_offset + wt["start"], 4),
                "end":   round(current_offset + wt["end"],   4),
            })

        line_files.append(tmp_path)
        if timings:
            current_offset += max(wt["end"] for wt in timings) + 0.25  # short pause between lines
        else:
            current_offset += 2.0

    # Concatenate line audio files
    if len(line_files) == 1:
        import shutil
        shutil.copy(line_files[0], output_path)
    else:
        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w") as f:
            for lf in line_files:
                f.write(f"file '{lf}'\n")
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", list_file, output_path],
            check=True, capture_output=True,
        )

    for lf in line_files:
        try:
            os.unlink(lf)
        except OSError:
            pass

    total_duration = current_offset
    return all_timings, total_duration


async def _build_all_scene_audios(
    scene_plans: list[dict], audio_dir: Path, voice: str, volume: str
) -> list[dict]:
    async def _one(scene: dict) -> dict:
        sid = scene["scene_id"]
        out_path = str(audio_dir / f"scene_{sid:02d}.mp3")
        print(f"  Generating voiceover for scene {sid}...")
        timings, duration = await _generate_scene_audio(
            scene=scene, voice=voice, volume=volume, output_path=out_path,
        )
        return {
            "scene_id":     sid,
            "audio_path":   out_path,
            "word_timings": timings,
            "duration":     round(duration, 3),
        }

    # All scenes in parallel — each is independent edge-tts I/O
    results = await asyncio.gather(*[_one(scene) for scene in scene_plans])
    # Maintain scene order regardless of completion order
    return sorted(results, key=lambda x: x["scene_id"])


def voiceover_node(state: ShortsState) -> dict:
    cfg        = state["cfg"]
    tts_cfg    = cfg["tts"]
    scene_plans = state.get("scene_plans", [])
    topic_slug = state["topic"][:30].lower().replace(" ", "_").replace("/", "_")

    if not scene_plans:
        return {"errors": ["voiceover_agent: no scene plans available"]}

    audio_dir = Path(cfg["video"]["output_dir"]) / topic_slug / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene_audios = asyncio.run(_build_all_scene_audios(
            scene_plans=scene_plans,
            audio_dir=audio_dir,
            voice=tts_cfg.get("voice", "en-US-AriaNeural"),
            volume=tts_cfg.get("volume", "+0%"),
        ))
        total_words = sum(len(sa["word_timings"]) for sa in scene_audios)
        total_dur   = sum(sa["duration"] for sa in scene_audios)
        print(f"  Voiceover done — {len(scene_audios)} scenes, {total_words} words, {total_dur:.1f}s total")
        return {"scene_audios": scene_audios}
    except Exception as e:
        return {"errors": [f"voiceover_agent: {e}"]}
