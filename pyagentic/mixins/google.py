from typing import Any
import requests
from pyagentic import spec, Tool, Event, Policy


class SearchMemoryPolicy(Policy):
    """
    Keeps the agent's search history trimmed to the latest 10 results.
    Automatically invoked after each search.
    """

    def __init__(self, memory_limit: int = 10):
        self.memory_limit = memory_limit

    def handle_event(self, event: Event, agent: "GoogleSearchMixin") -> None:
        if event.name == "search_performed":
            history = agent.search_history
            if len(history) > self.memory_limit:
                # keep only the 10 most recent searches
                agent.search_history = history[-10:]


class GoogleSearchMixin:
    """
    A Mixin that provides Google Search capability to an agent.

    Uses the Google Custom Search JSON API:
    https://developers.google.com/custom-search/v1/overview
    """

    google_api_key: spec.State[str] = spec.State(default=None)
    google_cx_id: spec.State[str] = spec.State(default=None)
    search_history: spec.State[list[dict[str, Any]]] = spec.State(
        default_factory=list, policies=[SearchMemoryPolicy(memory_limit=25)]
    )

    @Tool
    def google_search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """
        Perform a Google Search and return structured results.
        """
        if not self.google_api_key or not self.google_cx_id:
            raise ValueError(
                "GoogleSearchMixin requires `google_api_key` and `google_cx_id` to be set."
            )

        params = {
            "q": query,
            "cx": self.google_cx_id,
            "key": self.google_api_key,
            "num": num_results,
            "lr": f"lang_{language}",
        }

        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        resp.raise_for_status()
        data = resp.json()

        results = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "displayLink": item.get("displayLink"),
            }
            for item in data.get("items", [])
        ]

        # append new search results
        self.search_history.append({"query": query, "results": results})

        # trigger memory policy trimming
        event = Event(name="search_performed", payload={"query": query})
        for policy in getattr(self, "__policies__", []):
            policy.handle_event(event, self)

        return results
