from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from providers.llm_factory import get_llm
from graph.state import ShortsState


class ResearchResponse(BaseModel):
    summary: str = Field(description="Concise research summary within the word limit")
    key_facts: list[str] = Field(description="3-5 surprising or little-known facts/statistics")
    target_audience: str = Field(description="Who is this for? e.g. 'students 15-25 curious about AI'")
    unique_angle: str = Field(description="The most compelling angle or hook for a Short on this topic")
    common_misconceptions: list[str] = Field(description="1-2 myths to bust or pitfalls to avoid")


RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert researcher and fact-checker for viral educational content.

Rules:
- Summary must stay within the word limit — every word must earn its place
- Prioritise facts that will SURPRISE the viewer or challenge assumptions
- Identify the angle that makes someone stop scrolling
- Be specific: use numbers, names, dates where possible
- Flag any common myths about this topic the script should avoid"""),
    ("human", """Research this topic for a 60-second YouTube Short.

Topic: {topic}
Maximum summary words: {max_words}

Provide a concise research brief."""),
])


def research_node(state: ShortsState) -> dict:
    cfg = state["cfg"]
    llm = get_llm(cfg["agents"]["research"])
    chain = RESEARCH_PROMPT | llm.with_structured_output(ResearchResponse)
    max_words = cfg.get("research", {}).get("max_words", 100)

    try:
        print("  Researching topic...")
        result: ResearchResponse = chain.invoke({
            "topic": state["topic"],
            "max_words": max_words,
        })
        return {
            "research_summary": result.summary,
            "key_facts": result.key_facts,
            "target_audience": result.target_audience,
            "unique_angle": result.unique_angle,
        }
    except Exception as e:
        return {"errors": [f"research_agent: {e}"]}
