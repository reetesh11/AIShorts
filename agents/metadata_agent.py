from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class MetadataResponse(BaseModel):
    title: str = Field(description="The single best title, under 60 characters")
    title_options: list[str] = Field(description="3 click-worthy title options, under 60 chars each")
    description: str = Field(
        description="Full YouTube description (150-250 words). Include: "
                    "engaging opening sentence, key points covered, why this matters, "
                    "CTA to follow. SEO-optimised naturally."
    )
    hashtags: list[str] = Field(description="6-8 hashtags with # prefix (mix of broad and niche)")
    thumbnail_prompt: str = Field(
        description="Image generation prompt for a high-CTR YouTube thumbnail. "
                    "Should include: bold text overlay suggestion, face/emotion if applicable, "
                    "high contrast colors, clear subject."
    )


METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a YouTube SEO expert and viral content strategist.
You specialise in YouTube Shorts metadata that drives views, clicks, and subscribers.

Title rules:
- Under 60 characters
- Use numbers, power words, curiosity gaps, "how", "why", "what"
- Never clickbait — deliver on the promise
- Test different angles: question / statistic / bold claim

Description rules:
- First 2 lines show before "Show more" — make them count
- Include natural keyword variations
- Mention specific value: what the viewer will learn
- End with CTA

Hashtag rules:
- Always include #shorts and #youtubeshorts
- Mix: 2 broad (#science, #space), 4-5 niche-specific
- 6-8 total

Thumbnail prompt:
- YouTube thumbnails need: bold readable text, expressive face/reaction, bright contrasting colors
- Describe a thumbnail that would get 10%+ CTR"""),
    ("human", """Topic: {topic}
Target audience: {target_audience}
Key facts: {key_facts}
Script hook: {hook}
Research angle: {unique_angle}

Generate complete YouTube metadata."""),
])


def metadata_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    llm = get_llm(cfg["agents"]["metadata"])
    chain = METADATA_PROMPT | llm.with_structured_output(MetadataResponse)

    try:
        print("  Generating metadata...")
        result: MetadataResponse = chain.invoke({
            "topic": state["topic"],
            "target_audience": state.get("target_audience", "general audience"),
            "key_facts": "\n".join(f"- {f}" for f in state.get("key_facts", [])),
            "hook": state.get("hook", ""),
            "unique_angle": state.get("unique_angle", ""),
        })
        return {
            "metadata": {
                "title":           result.title,
                "title_options":   result.title_options,
                "description":     result.description,
                "hashtags":        result.hashtags,
                "thumbnail_prompt": result.thumbnail_prompt,
            }
        }
    except Exception as e:
        return {"errors": [f"metadata_agent: {e}"]}
