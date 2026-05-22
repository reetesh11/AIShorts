#!/usr/bin/env python3
"""
YouTube Shorts Agent — generate a complete 60s Short from a topic.

Usage:
    python main.py "The speed of light explained simply"
    python main.py --config config/config.yaml "Black holes for beginners"
    python main.py --dry-run "Quantum computing basics"
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_token_summary():
    from providers.llm_factory import get_token_usage
    usage = get_token_usage()
    if not usage["calls"]:
        return
    print("\n── Token Usage ──────────────────────────────────────")
    per_call_labels = [
        "research", "script", "quality_gate", "screenplay",
        "image_prompt", "metadata",
    ]
    for i, rec in enumerate(usage["per_call"]):
        label = per_call_labels[i] if i < len(per_call_labels) else f"call_{i+1}"
        print(f"  {label:<14} in:{rec['input']:>5}  out:{rec['output']:>4}  total:{rec['total']:>5}")
    print(f"  {'TOTAL':<14} in:{usage['input_tokens']:>5}  out:{usage['output_tokens']:>4}  total:{usage['total_tokens']:>5}")
    print("─────────────────────────────────────────────────────")


def print_results(state: dict):
    print("\n" + "=" * 60)
    print("SHORTS GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nTOPIC: {state.get('topic', '')}")

    if state.get("research_summary"):
        print(f"\nRESEARCH SUMMARY:\n{state['research_summary'][:300]}...")

    if state.get("hook"):
        print(f"\nHOOK: {state['hook']}")

    if state.get("quality_approved") is not None:
        approved = state["quality_approved"]
        notes = state.get("quality_notes", "")
        print(f"\nQUALITY GATE: {'PASSED' if approved else 'FLAGGED'}")
        if notes:
            print(f"  Notes: {notes}")

    if state.get("art_style"):
        print(f"\nART STYLE: {state['art_style']}")

    print("\nSCREENPLAY:")
    for scene in state.get("scene_plans", []):
        sid = scene["scene_id"]
        dur = scene.get("duration_hint", 0)
        tx  = scene.get("transition_in", "dissolve")
        print(f"\n  Scene {sid} ({tx}, ~{dur:.0f}s)")
        print(f"  Visual: {scene.get('image_description', '')[:80]}")
        for line in scene.get("voiceover_lines", []):
            print(f"  [{line.get('tone','neutral').upper()}] {line.get('text','')}")

    print("\nIMAGE PROMPTS:")
    for ip in state.get("image_prompts", []):
        print(f"\n  Scene {ip.get('scene_id','?')} [{ip.get('style_type','')}]")
        print(f"  {ip.get('prompt','')[:100]}")

    metadata = state.get("metadata", {})
    if metadata:
        print("\nTITLE OPTIONS:")
        for t in metadata.get("title_options", [metadata.get("title", "")]):
            print(f"  • {t}")
        print("\nHASHTAGS:")
        print("  " + " ".join(metadata.get("hashtags", [])))

    if state.get("video_path"):
        print(f"\nVIDEO: {state['video_path']}")

    if state.get("output_dir"):
        print(f"OUTPUT DIR: {state['output_dir']}/")

    if state.get("errors"):
        print("\nWARNINGS/ERRORS:")
        for err in state["errors"]:
            print(f"  ⚠  {err}")

    print("\n" + "=" * 60)


def _merge(state: dict, update: dict) -> dict:
    for k, v in update.items():
        if k == "errors" and isinstance(v, list):
            state["errors"] = state.get("errors", []) + v
        else:
            state[k] = v
    return state


def _initial_state(topic: str, cfg: dict) -> dict:
    return {
        "topic": topic,
        "cfg": cfg,
        "errors": [],
        "generated_images": [],
        "scene_plans": [],
        "image_prompts": [],
        "scene_audios": [],
        "metadata": {},
    }


def main():
    parser = argparse.ArgumentParser(description="YouTube Shorts AI Agent")
    parser.add_argument("topic", help="Topic for the YouTube Short")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run text agents only (skip image gen, TTS, video assembly)",
    )
    parser.add_argument(
        "--output-json",
        help="Save full state to a JSON file",
    )
    args = parser.parse_args()

    cfg_raw = load_config(args.config)

    # Validate required API keys
    missing = []

    def needs_provider(cfg, provider):
        for agent in cfg.get("agents", {}).values():
            if agent.get("provider") == provider:
                return True
        if cfg.get("image_generation", {}).get("provider") == provider:
            return True
        return False

    if needs_provider(cfg_raw, "anthropic") and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if needs_provider(cfg_raw, "google") and not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if needs_provider(cfg_raw, "groq") and not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")

    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    if args.dry_run:
        _run_dry(args.topic, cfg_raw, args.output_json)
    else:
        _run_full(args.topic, cfg_raw, args.output_json)


def _run_full(topic: str, cfg: dict, output_json: str | None):
    from graph.pipeline import build_pipeline

    pipeline = build_pipeline()
    initial = _initial_state(topic, cfg)

    print(f'\nGenerating Short for: "{topic}"')
    print("Pipeline: research → script → quality_gate → screenplay")
    print("          → [image_prompt → image_gen] ∥ voiceover ∥ metadata")
    print("          → stitch → output\n")

    step_labels = {
        "research":     "Researching topic...",
        "script":       "Writing script...",
        "quality_gate": "Checking quality...",
        "screenplay":   "Planning screenplay...",
        "image_prompt": "Building image prompts...",
        "image_gen":    "Generating images...",
        "voiceover":    "Synthesising voiceover...",
        "metadata":     "Generating metadata...",
        "stitch":       "Stitching video...",
        "output":       "Saving outputs...",
    }

    state = {k: v for k, v in initial.items()}
    for event in pipeline.stream(initial):
        for node_name, node_output in event.items():
            label = step_labels.get(node_name, node_name)
            print(f"[{node_name}] {label}")
            if isinstance(node_output, dict):
                _merge(state, node_output)

    print_results(state)
    print_token_summary()

    if output_json:
        _save_json(state, output_json)


def _run_dry(topic: str, cfg: dict, output_json: str | None):
    """Text-only dry run: research → script → quality_gate → screenplay → image_prompts → metadata."""
    from agents.research_agent     import research_node
    from agents.script_agent       import script_node
    from agents.quality_gate_agent import quality_gate_node
    from agents.screenplay_agent   import screenplay_node
    from agents.image_prompt_agent import image_prompt_node
    from agents.metadata_agent     import metadata_node

    state = _initial_state(topic, cfg)

    steps = [
        ("research",     research_node,     "Researching topic..."),
        ("script",       script_node,       "Writing script..."),
        ("quality_gate", quality_gate_node, "Checking quality..."),
        ("screenplay",   screenplay_node,   "Planning screenplay..."),
        ("image_prompt", image_prompt_node, "Building image prompts..."),
        ("metadata",     metadata_node,     "Generating metadata..."),
    ]

    print(f'\n[DRY RUN] Generating Short for: "{topic}"\n')
    for name, fn, label in steps:
        print(f"[{name}] {label}")
        _merge(state, fn(state))

    state.setdefault("video_path", "")
    state.setdefault("output_dir", "")

    print_results(state)
    print_token_summary()

    if output_json:
        _save_json(state, output_json)


def _save_json(state: dict, path: str):
    safe = {k: v for k, v in state.items() if k != "cfg"}
    Path(path).write_text(json.dumps(safe, indent=2, default=str))
    print(f"State saved to {path}")


if __name__ == "__main__":
    main()
