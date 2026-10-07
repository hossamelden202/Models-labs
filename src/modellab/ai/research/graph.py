from langgraph.graph import END, START, StateGraph

from modellab.ai.research.nodes import make_nodes
from modellab.ai.research.state import ResearchState


def _after_context(state):
    return "go" if state.get("menu") else "stop"


def build_graph(root, llm, knowledge):
    nodes = make_nodes(root, llm, knowledge)
    graph = StateGraph(ResearchState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "collect_context")
    graph.add_conditional_edges("collect_context", _after_context, {"go": "analyze_problem", "stop": END})
    for a, b in [
        ("analyze_problem", "retrieve_knowledge"),
        ("retrieve_knowledge", "generate_hypothesis"),
        ("generate_hypothesis", "generate_experiment"),
        ("generate_experiment", "validate_experiment"),
        ("validate_experiment", END),
    ]:
        graph.add_edge(a, b)
    return graph.compile()


def run_research(root, family_id, question, llm, knowledge):
    state = build_graph(root, llm, knowledge).invoke({"request": question, "family_id": family_id})
    return {
        "family_id": family_id,
        "question": question,
        "llm_used": state.get("llm_used", False),
        "failure_source": state.get("failure_source"),
        "analysis": state.get("analysis"),
        "retrieved": [{k: r[k] for k in ("id", "title", "score")} for r in state.get("retrieved", [])],
        "ranking": state.get("ranking", []),
        "hypothesis": state.get("hypothesis"),
        "proposal": state.get("proposal"),
        "validation": state.get("validation"),
        "notes": state.get("notes", []),
        "trace": state.get("trace", []),
    }
