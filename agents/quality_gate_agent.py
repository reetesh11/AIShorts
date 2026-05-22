from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class QualityResponse(BaseModel):
    approved: bool = Field(description="True if the script is good enough to proceed")
    hook_score: int = Field(description="Hook quality score 1-10")
    pacing_score: int = Field(description="Script pacing score 1-10")
    cta_score: int = Field(description="CTA strength score 1-10")
    word_count: int = Field(description="Approximate word count of the full script")
    notes: str = Field(description="One sentence of actionable feedback or 'Looks great!' if approved")
    improved_hook: str = Field(
        description="A stronger alternative hook if hook_score < 7, else repeat original hook"
    )


QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a YouTube Shorts performance analyst.
Evaluate scripts strictly — most Shorts fail because of a weak hook or poor pacing.

Scoring rubric:
Hook (1-10):
  10 = Irresistible — viewer cannot scroll past
  7-9 = Strong — most viewers will watch
  4-6 = Weak — too generic, doesn't create urgency
  1-3 = Fails — viewer will scroll immediately

Pacing (1-10):
  10 = Every sentence drives forward, no filler
  7-9 = Good momentum, minor slack
  4-6 = Some filler or redundant sentences
  1-3 = Slow, reader loses interest

CTA (1-10):
  10 = Clear, specific, natural
  7-9 = Good but slightly forced
  4-6 = Too generic or missing
  1-3 = No CTA or multiple competing asks

Approve if all scores >= 6 AND word count is 100-160."""),
    ("human", """Evaluate this YouTube Shorts script:

Topic: {topic}
Hook: {hook}

Full script:
{full_script}

CTA: {cta}"""),
])


def quality_gate_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    # Quality gate is optional — skip if not configured
    if not cfg.get("agents", {}).get("quality_gate"):
        return {"quality_approved": True, "quality_notes": "Quality gate skipped (not configured)"}

    llm = get_llm(cfg["agents"]["quality_gate"])
    chain = QUALITY_PROMPT | llm.with_structured_output(QualityResponse)

    try:
        print("  Running quality check...")
        result: QualityResponse = chain.invoke({
            "topic": state["topic"],
            "hook": state.get("hook", ""),
            "full_script": state.get("full_script", ""),
            "cta": state.get("cta", ""),
        })

        print(f"  Quality → hook:{result.hook_score}/10  pacing:{result.pacing_score}/10  "
              f"cta:{result.cta_score}/10  words:{result.word_count}  approved:{result.approved}")

        # Use improved hook if the original was weak
        updates: dict = {
            "quality_approved": result.approved,
            "quality_notes": result.notes,
        }
        if result.hook_score < 7 and result.improved_hook:
            print(f"  Hook improved: {result.improved_hook[:80]}...")
            updates["hook"] = result.improved_hook
            # Also patch the full script — replace first sentence with improved hook
            script = state.get("full_script", "")
            first_period = script.find(".")
            if first_period > 0:
                updates["full_script"] = result.improved_hook + script[first_period:]

        return updates
    except Exception as e:
        # Non-fatal — let the pipeline continue
        return {"quality_approved": True, "quality_notes": f"Quality gate error (skipped): {e}"}
