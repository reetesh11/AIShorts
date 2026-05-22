"""
Output Agent — organises all artifacts into a per-topic folder and writes description.md.
"""

import json
from datetime import datetime
from pathlib import Path
from graph.state import ShortsState


def output_node(state: ShortsState) -> dict:
    topic      = state["topic"]
    topic_slug = topic[:30].lower().replace(" ", "_").replace("/", "_")
    cfg        = state["cfg"]

    output_dir = Path(cfg["video"]["output_dir"]) / topic_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata    = state.get("metadata", {})
    scene_plans = state.get("scene_plans", [])
    key_facts   = state.get("key_facts", [])
    image_prompts = state.get("image_prompts", [])

    # ── Build description.md ─────────────────────────────────────────────────
    lines = [
        f"# {topic}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Titles (pick one)",
        "",
    ]

    for i, t in enumerate(metadata.get("title_options", [metadata.get("title", topic)]), 1):
        lines.append(f"{i}. **{t}**")

    lines += [
        "",
        "## YouTube Description",
        "",
        metadata.get("description", ""),
        "",
        "## Hashtags",
        "",
        " ".join(metadata.get("hashtags", [])),
        "",
        "---",
        "",
        "## Script",
        "",
        f"**Hook:** {state.get('hook', '')}",
        "",
        "```",
        state.get("full_script", ""),
        "```",
        "",
        "---",
        "",
        "## Screenplay",
        "",
    ]

    for scene in scene_plans:
        sid   = scene["scene_id"]
        lines.append(f"### Scene {sid} — `{scene.get('transition_in', 'dissolve')}` — ~{scene.get('duration_hint', 0):.0f}s")
        lines.append("")
        lines.append(f"**Visual:** {scene.get('image_description', '')}")
        lines.append("")
        lines.append("**Voiceover:**")
        for line in scene.get("voiceover_lines", []):
            lines.append(f"- [{line.get('tone', 'neutral').upper()}] {line.get('text', '')}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Research — Key Facts",
        "",
    ]
    for fact in key_facts:
        lines.append(f"- {fact}")

    lines += [
        "",
        "## Research Summary",
        "",
        state.get("research_summary", ""),
        "",
        "---",
        "",
        "## Image Prompts Used",
        "",
    ]
    for ip in image_prompts:
        sid = ip.get("scene_id", "?")
        lines.append(f"**Scene {sid}** `[{ip.get('style_type', '')}]`")
        lines.append(f"> {ip.get('prompt', '')}")
        lines.append("")

    if metadata.get("thumbnail_prompt"):
        lines += [
            "## Thumbnail Prompt",
            "",
            f"> {metadata['thumbnail_prompt']}",
            "",
        ]

    if state.get("errors"):
        lines += [
            "---",
            "",
            "## Warnings",
            "",
        ]
        for err in state["errors"]:
            lines.append(f"- ⚠ {err}")

    md_path = output_dir / "description.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # ── Save a lightweight state snapshot (JSON) ──────────────────────────────
    snapshot = {
        "topic":    topic,
        "video":    state.get("video_path", ""),
        "title":    metadata.get("title", ""),
        "hashtags": metadata.get("hashtags", []),
        "generated_at": datetime.now().isoformat(),
    }
    (output_dir / "meta.json").write_text(json.dumps(snapshot, indent=2))

    print(f"  Output saved to: {output_dir}/")
    print(f"    ├── short.mp4")
    print(f"    ├── description.md")
    print(f"    ├── meta.json")
    print(f"    ├── audio/scene_*.mp3")
    print(f"    └── images/scene_*.png")

    return {"output_dir": str(output_dir)}
