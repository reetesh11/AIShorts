from langgraph.graph import StateGraph, END
from graph.state import ShortsState
from agents.research_agent     import research_node
from agents.script_agent       import script_node
from agents.quality_gate_agent import quality_gate_node
from agents.screenplay_agent   import screenplay_node
from agents.image_prompt_agent import image_prompt_node
from agents.image_gen_agent    import image_gen_node
from agents.voiceover_agent    import voiceover_node
from agents.metadata_agent     import metadata_node
from agents.stitch_agent       import stitch_node
from agents.output_agent       import output_node


def build_pipeline() -> StateGraph:
    workflow = StateGraph(ShortsState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("research",      research_node)
    workflow.add_node("script",        script_node)
    workflow.add_node("quality_gate",  quality_gate_node)
    workflow.add_node("screenplay",    screenplay_node)
    workflow.add_node("image_prompt",  image_prompt_node)
    workflow.add_node("image_gen",     image_gen_node)
    workflow.add_node("voiceover",     voiceover_node)
    workflow.add_node("metadata",      metadata_node)
    workflow.add_node("stitch",        stitch_node)
    workflow.add_node("output",        output_node)

    # ── Sequential pipeline ───────────────────────────────────────────────────
    # Image gen is the network bottleneck — sequential is simpler and equally fast.
    workflow.set_entry_point("research")
    workflow.add_edge("research",      "script")
    workflow.add_edge("script",        "quality_gate")
    workflow.add_edge("quality_gate",  "screenplay")
    workflow.add_edge("screenplay",    "image_prompt")
    workflow.add_edge("image_prompt",  "image_gen")
    workflow.add_edge("image_gen",     "voiceover")
    workflow.add_edge("voiceover",     "metadata")
    workflow.add_edge("metadata",      "stitch")
    workflow.add_edge("stitch",        "output")
    workflow.add_edge("output",        END)

    return workflow.compile()
