from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class ScriptResponse(BaseModel):
    hook: str = Field(
        description="The opening 3-second line — must be a scroll-stopper. "
                    "Use question, shocking stat, or 'what if' format."
    )
    full_script: str = Field(
        description="Complete 45-60 second script including hook, body, and CTA. "
                    "~120-150 words. Written to be spoken, not read."
    )
    cta: str = Field(
        description="Call-to-action at the end (follow, comment, share)."
    )


SCRIPT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a YouTube Shorts creator with millions of subscribers.
You know exactly what makes people stop scrolling and watch to the end.

Script rules:
- Hook: first line must make the viewer feel they CANNOT swipe past. Under 5 seconds.
  Use one of: shocking statistic, bold claim, direct question, 'what if' scenario.
- Body: deliver the promise of the hook. Build curiosity then satisfy it.
  Use "you" to address the viewer. Short punchy sentences. Conversational tone.
- Payoff: give the viewer something valuable — a fact they didn't know, a perspective shift.
- CTA: end with one clear action. Don't ask for multiple things.
- Total: 120-150 words, 45-60 seconds spoken at natural pace.
- Integrate the key facts naturally — don't just list them.
- No filler. Every sentence must earn its place."""),
    ("human", """Topic: {topic}
Target audience: {target_audience}
Unique angle: {unique_angle}
Key facts to weave in: {key_facts}
Research context: {research_summary}

Write the YouTube Shorts script."""),
])


def script_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    llm = get_llm(cfg["agents"]["script"])
    chain = SCRIPT_PROMPT | llm.with_structured_output(ScriptResponse)

    try:
        print("  Writing script...")
        result: ScriptResponse = chain.invoke({
            "topic": state["topic"],
            "target_audience": state.get("target_audience", "general audience"),
            "unique_angle": state.get("unique_angle", ""),
            "key_facts": "\n".join(f"- {f}" for f in state.get("key_facts", [])),
            "research_summary": state.get("research_summary", ""),
        })
        return {
            "hook": result.hook,
            "full_script": result.full_script,
            "cta": result.cta,
        }
    except Exception as e:
        return {"errors": [f"script_agent: {e}"]}
