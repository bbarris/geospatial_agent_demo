"""
llm_agent.py - Example LLM agent + tools wiring.

This is a minimal demo agent for exploration only.
"""

import json
from typing import Any, Dict

from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

from tools import search_stac, compute_ndvi_for_item


# ---------------------------------------------------------------------
# Wrap functions into agent Tools
# ---------------------------------------------------------------------

def _search_stac_wrapper(args_str: str) -> str:
    """
    Args arrive as a string; parse into python objects.
    Expected: args_str is JSON-like:
      {"bbox":[...], "max_items":3}
    """
    args = json.loads(args_str)
    result = search_stac(**args)
    return json.dumps(result)


def _compute_ndvi_wrapper(args_str: str) -> str:
    """
    Expected: {"item_id":"...", "clip_bbox":[...]}
    """
    args = json.loads(args_str)
    result = compute_ndvi_for_item(**args)
    return json.dumps(result)


tools = [
    Tool(
        name="search_stac",
        func=_search_stac_wrapper,
        description=(
            "Search Sentinel-2 items.\n"
            "Input JSON: {\"bbox\":[minx,miny,maxx,maxy], \"start_date\":\"YYYY-MM-DD\", "
            "\"end_date\":\"YYYY-MM-DD\", \"max_items\":5, \"cloud_lt\":30}"
        )
    ),
    Tool(
        name="compute_ndvi",
        func=_compute_ndvi_wrapper,
        description=(
            "Compute NDVI stats for an item.\n"
            "Input JSON: {\"item_id\":\"...\", \"clip_bbox\":[minx,miny,maxx,maxy]}"
        )
    ),
]


# ---------------------------------------------------------------------
# Create agent
# ---------------------------------------------------------------------

def make_agent(
    model_name="gpt-4o-mini",
    temperature=0
):
    """
    Create and return an agent with the registered tools.
    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
    )

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    return agent


# ---------------------------------------------------------------------
# Example driver
# ---------------------------------------------------------------------

def run_query(prompt: str, model_name="gpt-4o-mini") -> Dict[str, Any]:
    """
    Instantiate an agent and run a query.
    Useful for quick tests or demos.
    """
    agent = make_agent(model_name=model_name)
    result = agent.run(prompt)
    return result


if __name__ == "__main__":
    # simple smoke test:
    prompt = (
        "Search for Sentinel-2 scenes over bbox [-122.06,37.29,-121.88,37.42] "
        "and return 2 results."
    )
    print(run_query(prompt))
