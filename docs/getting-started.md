# Getting Started with PyAgentic

PyAgentic is a declarative framework for building AI agents with OpenAI integration. This guide will walk you through building a research assistant agent step by step, introducing each core concept along the way.

## Installation

First, install PyAgentic:

```bash
pip install pyagentic-core
```

You'll also need an OpenAI API key for this tutorial.

## Why PyAgentic?

Imagine you're a researcher who needs to organize papers, extract key insights, and generate summaries. You want an AI assistant that can:

- Remember what papers you've added to your collection
- Extract and summarize key findings from papers
- Answer questions about your research collection
- Maintain context across conversations

This is exactly the kind of stateful, tool-equipped agent that PyAgentic makes easy to build.

## Step 1: Your First Agent

Let's start by creating a simple research assistant agent:

```python
from pyagentic import Agent

class ResearchAgent(Agent):
    """An AI assistant for managing and analyzing research papers."""
    
    __system_message__ = """
    You are a research assistant that helps organize and analyze academic papers.
    You maintain a collection of research papers and can answer questions about them.
    You are knowledgeable, precise, and helpful in academic contexts.
    """
```

This creates a basic conversational agent, but it can't do much beyond chat. Let's add some memory and capabilities.

## Step 2: Adding Context with ContextItem

Real research assistants need to remember things! Let's add some persistent context to track the papers in our collection:

```python
from pyagentic import Agent, ContextItem
from typing import List, Dict
from collections import defaultdict

from arxiv import Result as Paper

class ResearchAgent(Agent):
    """An AI assistant for managing and analyzing research papers."""
    
    __system_message__ = """
    You are a research assistant that helps organize and analyze academic papers.
    You have full access to the Arxiv

    Available topics: {available_topics}
    
    Use your tools to help users manage and analyze their research collection.
    Feel free to use many tools at once
    """

    __input_template__ = """
    Current Topic: {current_topic}

    User Message: {user_message}
    """
    
    papers: Dict[str, List[Paper]] = ContextItem(default_factory=lambda: defaultdict(list))
    current_topic: str = ContextItem(default="General Research")
```

Now our agent has memory! The `papers` list and `current_topic` string persist between conversations. Notice how we reference `{available_topics}` in the system message - we'll make that work next.

## Step 3: Adding Tools for Actions

Let's give our agent the ability to actually search and papers by adding tools:

```python
@tool("Search the Arxiv for relevant papers")
def search(
    self,
    terms: list[str] = ParamInfo(required=True, description="Terms you think are relevant to the user's query. Be creative")
) -> str:
    print(f"Searching terms: {terms}")
    found = []
    for term in terms:
        results = search_arxiv(term)
        found.extend(results)
    return json.dumps(found, indent=2)

@tool("Add a new paper to the research collection")
def add_paper(
    self,
    paper_id: str = ParamInfo(required=True, description="ID of the paper from search results"),
    topic: str = ParamInfo(
        required=False,
        description="Research topic/area",
    )
) -> str:
    print(f"Adding {paper_id} to {topic}")
    paper = get_paper(paper_id)
    self.context.papers[topic].append(paper)
    return f"Added paper '{paper.title}' by {paper.authors} to your collection under topic '{topic}'."
```

Now our agent can perform actions! The `@tool` decorator exposes methods to the AI, and `ParamInfo` helps define parameter requirements and descriptions.

A problem may arise here, asking the LLM to generate raw paper ids may lead to hallucination...

## Step 4: Computed Context for Dynamic Values

This can be solved by passing over values to the param, ensuring the the LLM will only pick from that list.

To do so, we need to combine the use of a `computed_context` and a `ContextRef`

`computed_context`s work a lot like python `property`s, each time it is accessed, it is recalculated. We are going to add two here:
    - available_topics
    - paper_ids

Lastly, we can hook up these new context items using a `ContextRef`, ensuring that the string passed to the ref perfectly matches that of the context item name.
```python
@computed_context
def available_topics(self) -> List[str]:
    """Extract unique topics from paper summaries and titles"""

    return list(self.papers.keys())

@computed_context
def paper_ids(self) -> List[str]:
    """Get list of all paper titles for reference"""
    return [paper.get_short_id() for paper in self.papers[self.current_topic]]

@tool("Add a new paper to the research collection")
def add_paper(
    self,
    paper_id: str = ParamInfo(
        required=True, 
        description="ID of the paper from search results",
        values=ContextRef("paper_ids")
    ),
    topic: str = ParamInfo(
        required=False,
        description="Research topic/area",
        values=ContextRef("available_topics")  # Dynamic options!
    )
) -> str:
    print(f"Adding {paper_id} to {topic}")
    paper = get_paper(paper_id)
    self.context.papers[topic].append(paper)
    return f"Added paper '{paper.title}' by {paper.authors} to your collection under topic '{topic}'."
```

This will ensure that the LLM choices specific values when calling the tool.

## Step 5: Advanced Tool Parameters with ContextRef

What if we want our tools to be smarter about the data they work with? Let's use `ContextRef` to create dynamic parameter constraints:

```python
from pyagentic import Agent, ContextItem, tool, ParamInfo, computed_context, ContextRef
from typing import List, Dict
from collections import defaultdict

from utils import read_paper, get_paper, search_arxiv
from arxiv import Result as Paper

class ResearchAgent(Agent):
    """An AI assistant for managing and analyzing research papers."""
    
    __system_message__ = """
    You are a research assistant that helps organize and analyze academic papers.
    You have full access to the Arxiv

    Available topics: {available_topics}
    
    Use your tools to help users manage and analyze their research collection.
    Feel free to use many tools at once
    """

    __input_template__ = """
    Current Topic: {current_topic}

    User Message: {user_message}
    """
    
    papers: Dict[str, List[Paper]] = ContextItem(default_factory=lambda: defaultdict(list))
    current_topic: str = ContextItem(default="General Research")
    
    
    @computed_context
    def available_topics(self) -> List[str]:
        """Extract unique topics from paper summaries and titles"""

        return list(self.papers.keys())
    
    @computed_context
    def paper_titles(self) -> List[str]:
        """Get list of all paper titles for reference"""
        return [paper.title for paper in self.papers[self.current_topic]]
    
    @tool("Search the Arxiv for relevant papers")
    def search(
        self,
        terms: list[str] = ParamInfo(required=True, description="Terms you think are relevant to the user's query. Be creative")
    ) -> str:
        print(f"Searching terms: {terms}")
        found = []
        for term in terms:
            results = search_arxiv(term)
            found.extend(results)
        return json.dumps(found, indent=2)

    @tool("Add a new paper to the research collection")
    def add_paper(
        self,
        paper_id: str = ParamInfo(required=True, description="ID of the paper from search results"),
        topic: str = ParamInfo(
            required=False,
            description="Research topic/area",
            values=ContextRef("available_topics")  # Dynamic options!
        )
    ) -> str:
        print(f"Adding {paper_id} to {topic}")
        paper = get_paper(paper_id)
        self.context.papers[topic].append(paper)
        return f"Added paper '{paper.title}' by {paper.authors} to your collection under topic '{topic}'."
    
    @tool("Read the entire content of a paper")
    def read_paper(
        self,
        title: str = ParamInfo(
            required=True,
            description="Exact title of the paper to get details for",
            values=ContextRef("paper_titles")  # Only allow existing paper titles!
        )
    ) -> str:
        print(f"Reading {title}")
        current_topic = self.context.current_topic
        for paper in self.context.papers[current_topic]:
            if title in paper.title:
                return read_paper(paper)
        return f"Paper '{title}' not found in collection."
    
    @tool("Update research focus topic, call this whenever you feel the subject has changed. Always call this before other tools if the subject has changed")
    def set_focus_topic(
        self,
        topic: str = ParamInfo(
            required=True,
            description="New research focus topic",
        )
    ) -> str:
        print(f"New focus: {topic}")
        self.context.current_topic = topic
        return f"Research focus updated to: {topic}"
```

Now our tools are much smarter! The `ContextRef` creates dynamic constraints:
- When adding papers, the `topic` parameter will only suggest topics that already exist in our collection
- When getting paper details, the agent can only choose from actual paper titles
- The available options update automatically as we add more papers

## Step 6: Putting It All Together

Let's see our complete research assistant in action:

```python
# Start with an empty collection
agent = ResearchAgent(
    model="gpt-4o",
    api_key="your-openai-api-key"
)

# Add some papers
await agent.run('''
Add this paper: "Attention Is All You Need" by Vaswani et al., 2017.
This paper introduced the Transformer architecture which revolutionized NLP.
''')

await agent.run('''
Add "BERT: Pre-training of Deep Bidirectional Transformers" by Devlin et al., 2018.
BERT showed how to effectively pre-train language models for downstream tasks.
''')

await agent.run('''
Add "Vision Transformer (ViT)" by Dosovitskiy et al., 2020.
This paper applied transformers directly to image classification.
''')

# Now the agent has intelligent constraints
response = await agent.run("Can you show me details about the Vision Transformer paper?")
print("Response:", response)

# Change focus
await agent.run("Set my research focus to Computer Vision")

# Ask for analysis
final_response = await agent.run(
    "What are the key connections between the papers in my collection? "
    "How did transformers evolve from NLP to computer vision?"
)
print("Analysis:", final_response)
```

## What Makes This Powerful

Our research assistant demonstrates all of PyAgentic's key strengths:

1. **Declarative Design**: We define what the agent should do, not how
2. **Persistent Context**: The agent remembers papers across conversations
3. **Dynamic Intelligence**: Tools adapt their constraints based on current data  
4. **Type Safety**: All parameters are properly typed and validated
5. **Natural Evolution**: Easy to add new capabilities without breaking existing functionality

## Key Concepts Summary

- **`Agent`**: Base class that handles OpenAI integration and orchestration
- **`ContextItem`**: Persistent state that survives between conversations
- **`@tool`**: Decorator that exposes methods as callable functions to the AI
- **`ParamInfo`**: Metadata for tool parameters (descriptions, requirements, defaults)  
- **`@computed_context`**: Dynamic properties that recalculate on each access
- **`ContextRef`**: Links tool parameters to live context data for smart constraints

## Next Steps

Now you're ready to build sophisticated AI agents! Try:

- Adding more complex tools (file I/O, API calls, data analysis)
- Creating multi-agent systems that collaborate
- Building domain-specific agents for your use cases
- Exploring advanced context patterns and validation

The declarative nature of PyAgentic makes it easy to iterate and extend your agents as your needs grow.
