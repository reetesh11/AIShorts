from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class VoiceoverLineModel(BaseModel):
    text: str = Field(description="Exact words to speak — one short punchy sentence")
    tone: str = Field(
        description="TTS delivery tone: excited | mysterious | dramatic | energetic | calm | warm | inspiring"
    )


class ScenePlanModel(BaseModel):
    scene_id: int = Field(description="Scene number starting from 1")
    image_description: str = Field(
        description="Specific visual description for AI image generation. "
                    "Include: subject, action, composition, lighting, mood, setting."
    )
    voiceover_lines: list[VoiceoverLineModel] = Field(
        description="1-3 short voiceover lines for this scene. Keep each line brief."
    )
    transition_in: str = Field(
        description="How this scene enters: dissolve (smooth) | cut (instant energy) | "
                    "fade (from black) | zoom_in (reveal) | zoom_out (wide reveal)"
    )
    duration_hint: float = Field(description="Target scene duration in seconds (5-15)")


class ScreenplayResponse(BaseModel):
    art_style: str = Field(
        description="One consistent art style for ALL images. "
                    "e.g. 'vibrant 3D cartoon style' or 'cinematic photorealistic style'"
    )
    scene_plans: list[ScenePlanModel] = Field(description="5-6 scene plans in order")


SCREENPLAY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an award-winning YouTube Shorts director.
Your job is to turn a script into a precise visual-audio scene plan.

Scene planning rules:
- Scene 1 = hook (≤5 seconds, visually shocking/beautiful to match the hook line)
- Scene 2-5 = body (build knowledge, maintain energy)
- Final scene = CTA (warm, direct, human)
- TOTAL duration across all scenes = 45-60 seconds
- Each scene has 1-3 SHORT voiceover lines (not long paragraphs)
- Assign tone PER LINE based on emotional feel of that specific sentence
- Image descriptions must be cinematic and specific enough for an AI image generator
- Choose transitions that serve the content:
    dissolve = calm, thoughtful moments
    cut      = high energy, fast reveals, statistics
    fade     = emotional weight, topic shifts
    zoom_in  = building suspense, reveals
    zoom_out = establishing context, big picture
- Art style MUST be consistent across all scenes"""),
    ("human", """Research context:
{research_summary}

Full script to adapt into scenes:
{full_script}

Create the scene-by-scene screenplay plan."""),
])


def screenplay_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    llm = get_llm(cfg["agents"]["screenplay"])
    chain = SCREENPLAY_PROMPT | llm.with_structured_output(ScreenplayResponse)

    try:
        print("  Planning screenplay...")
        result: ScreenplayResponse = chain.invoke({
            "research_summary": state.get("research_summary", ""),
            "full_script": state.get("full_script", ""),
        })
        return {
            "art_style": result.art_style,
            "scene_plans": [s.model_dump() for s in result.scene_plans],
        }
    except Exception as e:
        return {"errors": [f"screenplay_agent: {e}"]}
