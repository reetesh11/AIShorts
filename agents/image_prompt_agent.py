from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class ImagePromptModel(BaseModel):
    scene_id: int
    prompt: str = Field(
        description="Complete, detailed image generation prompt. "
                    "Must start with the art style. Include subject, composition, "
                    "lighting, mood, colors, camera angle."
    )
    style_type: str = Field(
        description="Primary style: 3d_cartoon | photorealistic | illustration | flat_design | cinematic"
    )


class ImagePromptsResponse(BaseModel):
    prompts: list[ImagePromptModel]


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at writing prompts for AI image generators (Stable Diffusion / FLUX / Imagen).
You create prompts that produce stunning, scroll-stopping visuals for YouTube Shorts.

Prompt rules:
- Start EVERY prompt with the art style phrase exactly as given
- Portrait/vertical composition only (9:16 frame)
- Include: subject, action, composition, lighting quality, mood/atmosphere, color palette, camera angle
- Be cinematic and specific — vague prompts produce generic images
- No text, words, letters, logos, or watermarks in the description
- Each scene should visually contrast slightly with the previous for visual variety
- Match the emotional tone of the scene's voiceover"""),
    ("human", """Art style for ALL images: {art_style}

Scenes to illustrate:
{scenes_text}

Generate one detailed image prompt per scene."""),
])


def image_prompt_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    llm = get_llm(cfg["agents"]["image_prompts"])
    chain = PROMPT_TEMPLATE | llm.with_structured_output(ImagePromptsResponse)

    scene_plans = state.get("scene_plans", [])
    art_style = state.get("art_style", "")

    if not scene_plans or not art_style:
        return {"errors": ["image_prompt_agent: no scene plans or art style available"]}

    vid_cfg = cfg["video"]
    width, height = vid_cfg["width"], vid_cfg["height"]

    scenes_text = "\n\n".join(
        f"Scene {s['scene_id']}:\n"
        f"  Visual: {s['image_description']}\n"
        f"  Mood: {', '.join(l['tone'] for l in s['voiceover_lines'])}"
        for s in scene_plans
    )

    try:
        print("  Crafting image prompts...")
        result: ImagePromptsResponse = chain.invoke({
            "art_style": art_style,
            "scenes_text": scenes_text,
        })

        prompts = [
            {
                "scene_id": p.scene_id,
                "prompt": p.prompt,
                "style_type": p.style_type,
                "width": width,
                "height": height,
            }
            for p in result.prompts
        ]
        return {"image_prompts": prompts}
    except Exception as e:
        return {"errors": [f"image_prompt_agent: {e}"]}
