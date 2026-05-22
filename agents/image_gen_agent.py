import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

from graph.state import ShortsState

_POLLINATIONS_MAX_RETRIES = 5
_POLLINATIONS_BACKOFF_BASE = 10   # seconds; doubles each retry
_POLLINATIONS_WORKERS = 3         # parallel image requests
_POLLINATIONS_START_STAGGER = 2.0 # seconds between each worker's first request

_NEGATIVE_PROMPT = "text,words,letters,watermark,logo,signature,blurry,distorted,deformed,ugly,low quality"


def _generate_pollinations(prompt: str, img_path: Path, width: int, height: int, stagger: float = 0.0):
    """Generate one image via Pollinations.ai (FLUX). Retries with backoff on 429."""
    if stagger > 0:
        time.sleep(stagger)

    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        f"?width={width}&height={height}"
        f"&nologo=true&model=flux&enhance=true"
        f"&negative={quote(_NEGATIVE_PROMPT)}"
        f"&seed={abs(hash(prompt)) % 99999}"
    )

    delay = _POLLINATIONS_BACKOFF_BASE
    for attempt in range(1, _POLLINATIONS_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 429:
                wait = delay + (attempt * 2)
                print(f"    Rate limited — waiting {wait}s (attempt {attempt}/{_POLLINATIONS_MAX_RETRIES})...")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            img_path.write_bytes(resp.content)
            return
        except requests.exceptions.HTTPError:
            if attempt == _POLLINATIONS_MAX_RETRIES:
                raise
            print(f"    HTTP error — retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Pollinations.ai failed after {_POLLINATIONS_MAX_RETRIES} retries")


def _generate_imagen(prompt: str, img_path: Path, model: str, aspect_ratio: str):
    """Gemini Imagen 3 — requires GOOGLE_API_KEY with billing enabled."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=genai_types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
        ),
    )
    if response.generated_images:
        img_path.write_bytes(response.generated_images[0].image.image_bytes)
    else:
        raise RuntimeError("Imagen returned no images")


def image_gen_node(state: ShortsState) -> dict:
    cfg      = state["cfg"]
    img_cfg  = cfg["image_generation"]
    provider = img_cfg.get("provider", "pollinations").lower()
    prompts  = state.get("image_prompts", [])
    topic_slug = state["topic"][:30].lower().replace(" ", "_").replace("/", "_")

    vid_cfg = cfg["video"]
    width, height = vid_cfg["width"], vid_cfg["height"]

    output_dir = Path(cfg["video"]["output_dir"]) / topic_slug / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prompts:
        return {"errors": ["image_gen_agent: no prompts available"]}

    errors: list[str] = []

    if provider == "pollinations":
        results: list[str | None] = [None] * len(prompts)

        def _run(idx: int, ip: dict) -> tuple[int, str]:
            prompt_text = ip["prompt"] if isinstance(ip, dict) else ip
            scene_id    = ip.get("scene_id", idx + 1) if isinstance(ip, dict) else idx + 1
            img_w       = ip.get("width", width)  if isinstance(ip, dict) else width
            img_h       = ip.get("height", height) if isinstance(ip, dict) else height
            img_path    = output_dir / f"scene_{scene_id:02d}.png"
            stagger     = idx * _POLLINATIONS_START_STAGGER
            print(f"  Generating image {idx + 1}/{len(prompts)} [pollinations] (scene {scene_id})...")
            _generate_pollinations(prompt_text, img_path, img_w, img_h, stagger)
            return idx, str(img_path)

        with ThreadPoolExecutor(max_workers=_POLLINATIONS_WORKERS) as executor:
            future_map = {executor.submit(_run, i, ip): i for i, ip in enumerate(prompts)}
            for future in as_completed(future_map):
                i = future_map[future]
                try:
                    idx, path = future.result()
                    results[idx] = path
                except Exception as e:
                    errors.append(f"image_gen_agent [image {i + 1}]: {e}")

        generated_paths = [p for p in results if p is not None]

    else:
        # Google Imagen: sequential (quota-conscious)
        generated_paths = []
        for i, ip in enumerate(prompts):
            prompt_text = ip["prompt"] if isinstance(ip, dict) else ip
            scene_id    = ip.get("scene_id", i + 1) if isinstance(ip, dict) else i + 1
            img_path    = output_dir / f"scene_{scene_id:02d}.png"

            print(f"  Generating image {i + 1}/{len(prompts)} [google] (scene {scene_id})...")
            try:
                _generate_imagen(
                    prompt_text, img_path,
                    model=img_cfg.get("model", "imagen-3.0-generate-002"),
                    aspect_ratio=img_cfg.get("aspect_ratio", "9:16"),
                )
                generated_paths.append(str(img_path))
            except Exception as e:
                errors.append(f"image_gen_agent [image {i + 1}]: {e}")

            if i < len(prompts) - 1:
                time.sleep(1)

    result: dict = {"generated_images": generated_paths}
    if errors:
        result["errors"] = errors
    return result
