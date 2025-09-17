# Getting Started with PyAgentic

PyAgentic is a declarative framework for building AI agents with support for multiple LLM providers including OpenAI, Anthropic, and others. This guide will walk you through building a research assistant agent step by step, introducing each core concept along the way.

## Installation

First, install PyAgentic:

```bash
pip install pyagentic-core
```

You'll also need an API key from your chosen LLM provider (OpenAI, Anthropic, etc.) for this tutorial.

## Why PyAgentic?

Imagine you're a researcher who needs to organize papers, extract key insights, and generate summaries. You want an AI assistant that can:

- Remember what papers you've added to your collection
- Extract and summarize key findings from papers
- Answer questions about your research collection
- Maintain context across conversations

This is exactly the kind of stateful, tool-equipped agent that PyAgentic makes easy to build.

## Choosing Your LLM Provider

PyAgentic supports multiple LLM providers out of the box. You can configure your agent to use any supported provider:

### Supported Providers

- **OpenAI**: GPT-4, GPT-3.5, and other OpenAI models
- **Anthropic**: Claude models with full tool calling support
- **Mock**: For testing and development without API costs

### Provider Configuration Methods

You can configure providers in two ways:

**Model String Format**
```python
agent = MyAgent(
    model="<provider>::<model_name>",
    api_key="your_api_key"
)

openai_agent = MyAgent(
    model="openai::gpt-5",
    api_key="your_api_key"
)

anthropic_agent = MyAgent(
    model="anthropic::claude-opus-4-1-20250805",
    api_key="your_api_key"
)
```

**Provider Instance**
```python
from pyagentic.llm import OpenAIProvider, AnthropicProvider

# OpenAI
agent = MyAgent(
    provider=OpenAIProvider(
        model="gpt-5",
        api_key="your_openai_key",
        max_retries=10,
        timeout=5
    )
)

# Anthropic
agent = MyAgent(
    provider=AnthropicProvider(
        model="claude-opus-4-1-20250805",
        api_key="your_anthropic_key",
        base_url="https://my-deployment.com/models"
    )
)
```

The provider instance method gives you more control over client configuration, allowing you to pass additional parameters like `base_url`, `timeout`, `max_retries`, etc.

## Step 1: Your First Agent

Let's start by creating a simple research assistant agent:

``` py linenums="1"
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

``` py linenums="1"
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

``` py linenums="1"
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

``` py linenums="1"
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

Let's put it all together while adding a couple more features to make it more complete, like:
    - Setting the focus
    - Read a paper

``` py linenums="1"
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

    @computed_context
    def paper_ids(self) -> List[str]:
        """Get list of all paper titles for reference"""
        return [paper.get_short_id() for paper in self.papers[self.current_topic]]

    
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

## Step 6: Let's run it!

First, we have to create our agent.

``` py linenums="1"
# Option 1: Using model string format
agent = ResearchAgent(
    model="openai::gpt-4o",
    api_key=API_KEY
)

# Option 2: Using a provider instance
from pyagentic.llm import OpenAIProvider

agent = ResearchAgent(
    provider=OpenAIProvider(
        model="gpt-4o",
        api_key=API_KEY
    )
)
```

Now lets run a couple messages through to see if it is working?

=== "Set Focus"
    ``` py linenums="1"
    await agent('''
    Im trying to find a link between ai usage and climate change
    ''')
    ```

    ```
    New focus: AI and Climate Change
    ```

    Let's explore research papers linking AI usage and climate change. I'll look for recent studies on this topic. Please hold on for a moment.Here are some research areas and findings regarding the connection between AI usage and climate change:

    1. **AI in Climate Modeling:**
    AI helps improve climate models by enhancing predictions of weather patterns, temperature changes, and extreme events like hurricanes. Machine learning algorithms can analyze vast datasets to improve climate forecasts.

    2. **Monitoring and Reducing Emissions:**
    AI aids in monitoring industrial emissions and optimizing energy consumption in various sectors, potentially reducing overall carbon footprints. Smart grids and AI-enhanced energy management systems help in efficient resource usage.

    3. **Environmental Data Analysis:**
    - AI processes environmental data from satellites and sensors to monitor deforestation, glacial melting, and other climate-related changes. This helps in providing timely insights for environmental protection efforts.

    4. **Sustainable Practices:**
    Machine learning supports the development of sustainable agricultural practices by analyzing crop yield data and predicting suitable planting strategies, reducing waste and resource usage.

    5. **AI's Carbon Footprint:**
    Training AI models is energy-intensive, contributing to carbon emissions. Research is exploring more efficient algorithms and hardware to mitigate these effects.

    Would you like to dive deeper into any specific area or look for more detailed research papers?

=== "Find Papers"
    ``` py linenums="1"
    response = await agent('''
    Ya! Find some papers around AI's Carbon Footprint
    ''')
    ```

    ```
    Searching terms: ['AI carbon footprint', 'energy consumption AI', 'AI environmental impact']
    ```

    Here are some papers related to AI's carbon footprint:

    1. **Integrating AI's Carbon Footprint into Risk Management Frameworks: Strategies and Tools for Sustainable Compliance in Banking Sector**  
    *Author: Nataliya Tkachenko*  
    This paper explores integrating AI’s carbon footprint into risk management in the banking sector, highlighting advancements like energy-efficient models and green cloud computing.  
    [Read more](https://arxiv.org/abs/2410.01818v1)

    2. **Carbon-Aware End-to-End Data Movement**  
    *Authors: Jacob Goldverg, Hasibul Jamil, Elvis Rodriguez, Tevfik Kosar*  
    Investigates carbon emissions from data movement in cloud computing and proposes optimization strategies for reducing energy consumption.  
    [Read more](https://arxiv.org/abs/2406.09650v1)

    3. **Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models**  
    *Authors: Pengfei Li, Jianyi Yang, Mohammad A. Islam, Shaolei Ren*  
    Tackles AI's overlooked water consumption, highlighting the necessity to address both water and carbon footprints for sustainable AI.  
    [Read more](https://arxiv.org/abs/2304.03271v5)

    4. **Towards Environmentally Equitable AI via Geographical Load Balancing**  
    *Authors: Pengfei Li, Jianyi Yang, Adam Wierman, Shaolei Ren*  
    Proposes equity-aware geographical load balancing to address regional disparities in AI's environmental impact.  
    [Read more](https://arxiv.org/abs/2307.05494v2)

    5. **CAFE: Carbon-Aware Federated Learning in Geographically Distributed Data Centers**  
    *Authors: Jieming Bian, Lei Wang, Shaolei Ren, Jie Xu*  
    Introduces a framework for carbon-aware federated learning to optimize training across distributed data centers while considering carbon intensity.  
    [Read more](https://arxiv.org/abs/2311.03615v2)

    If you want to explore any specific paper in detail, let me know!

=== "Add Papers"
    ``` py linenums="1"
    response = await agent('''
    Add the papers that you think are the most relevant to linking AI and carbon
    ''')
    ```

    ```
    Adding 2410.01818v1 to AI and Climate Change
    Adding 2406.09650v1 to AI and Climate Change
    Adding 2304.03271v5 to AI and Climate Change
    Adding 2407.05176v1 to AI and Climate Change
    ```

    I've added the most relevant papers connecting AI and carbon emissions to your collection. If you need further assistance or wish to explore more papers, feel free to ask!

=== "Analyze"
    ``` py linenums="1"
    response = await agent('''
    Analyze these papers and tell me what you learned with citiations
    ''')
    ```

    ```
    Reading Integrating AI's Carbon Footprint into Risk Management Frameworks: Strategies and Tools for Sustainable Compliance in Banking Sector
    Reading Carbon-Aware End-to-End Data Movement
    Reading Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models
    Reading Towards Socially and Environmentally Responsible AI
    ```

    Here's a summary and analysis of the selected papers linking AI and carbon:

    1. **Integrating AI's Carbon Footprint into Risk Management Frameworks** (Tkachenko, 2024)

    **Overview:** This paper discusses how AI's carbon footprint can be integrated into the banking sector's risk management frameworks (RMFs). It highlights the regulatory requirements driving financial institutions to consider environmental impacts within AI governance.
    **Key Insights:** AI models in banking require significant energy, contributing to carbon emissions. Efficient AI models and green cloud computing practices are recommended strategies to mitigate these impacts.
    **Importance:** Integrating AI's carbon footprint into RMFs can help banks comply with sustainability regulations and demonstrate environmental responsibility.

    2. **Carbon-Aware End-to-End Data Movement** (Goldverg et al., 2024)

    **Overview:** This study focuses on measuring and optimizing the carbon footprint of data movement across networks, a significant contributor to AI-related carbon emissions.
    **Key Insights:** The paper introduces carbon-aware scheduling strategies that optimize data transfer times and routes to reduce carbon emissions. It highlights the need for detailed monitoring of network paths and end systems.
    **Importance:** Addressing the carbon footprint in data movement is crucial for reducing the total carbon emissions associated with AI workloads.

    3. **Making AI Less "Thirsty"** (Li et al., 2025)

    **Overview:** This paper uncovers the water footprint of AI models, emphasizing that AI's environmental impact extends beyond carbon emissions to significant water consumption, particularly in data center cooling.
    **Key Insights:** The water consumption associated with AI could become a major ecological challenge. The study presents methodologies to estimate and reduce AI's water footprint.
    **Importance:** Highlighting AI's water usage as a crucial aspect of its environmental footprint encourages more holistic sustainability strategies.

    4. **Towards Socially and Environmentally Responsible AI** (Li et al., 2024)

    **Overview:** The paper proposes equitable geographical load balancing (GLB) to address AI's social and environmental inequalities, ensuring fair distribution of AI's benefits and burdens across regions.
    **Key Insights:** The disparities in AI's resource allocation can lead to environmental and social inequity. Implementing equity-aware GLB can balance AI's negative impacts more fairly across regions.
    **Importance:** By promoting social and environmental equity, this approach aims to foster responsible AI deployment.

    ### Conclusion

    The research underscores the multifaceted environmental impacts of AI, including carbon and water footprints, and stresses the importance of equitable resource distribution. The integration of these concerns into frameworks such as RMFs in banking and carbon-aware scheduling in data movement are vital steps towards sustainable AI practices.

If you want further analysis or details on specific papers, feel free to let me know!
## What Makes This Powerful

Our research assistant demonstrates all of PyAgentic's key strengths:

1. **Declarative Design**: We define what the agent should do, not how
2. **Persistent Context**: The agent remembers papers across conversations
3. **Dynamic Intelligence**: Tools adapt their constraints based on current data  
4. **Type Safety**: All parameters are properly typed and validated
5. **Natural Evolution**: Easy to add new capabilities without breaking existing functionality

## Controlling Tool Usage with max_call_depth

By default, PyAgentic agents can only call tools once per conversation turn, then must provide a final response. This prevents endless tool-calling loops but might limit complex workflows. You can control this behavior with the `max_call_depth` parameter:

```python
# Allow multiple rounds of tool calling
agent = ResearchAgent(
    model="openai::gpt-4o",
    api_key=API_KEY,
    max_call_depth=3  # Allow up to 3 rounds of tool calls
)
```

### How max_call_depth Works

- **Depth 0**: Agent can call tools, then must respond
- **Depth 1 (default)**: After tools execute, agent can call more tools or respond  
- **Depth 2+**: Agent can continue calling tools in multiple rounds

Each "depth" represents a full round of tool calling. Within each round, the agent can make multiple parallel tool calls, but after each round completes, it decides whether to call more tools or give a final answer.

### When to Increase max_call_depth

**Use higher depths (2-4) when your agent needs to:**
- Search for information, then analyze what it found
- Read multiple files and synthesize information  
- Perform multi-step research or analysis
- Chain tool outputs together

**Keep default (1) when your agent:**
- Has simple, single-purpose tools
- Should respond quickly without complex workflows
- Might get stuck in tool-calling loops

### Example: Research Agent with Multiple Depths

```
With max_call_depth=1 (default):
User: "Research AI and climate change" 
  → Agent calls search() → Responds with results

With max_call_depth=3:
User: "Research AI and climate change"
  → Depth 0: Agent calls search()  
  → Depth 1: Agent calls add_paper() for multiple papers
  → Depth 2: Agent calls read_paper() to analyze content
  → Final response with comprehensive analysis
```

This allows your agent to perform sophisticated multi-step workflows while preventing infinite loops.

## Key Concepts Summary

- **`Agent`**: Base class that handles LLM provider integration and orchestration
- **`ContextItem`**: Persistent state that survives between conversations
- **`@tool`**: Decorator that exposes methods as callable functions to the AI
- **`ParamInfo`**: Metadata for tool parameters (descriptions, requirements, defaults)  
- **`@computed_context`**: Dynamic properties that recalculate on each access
- **`ContextRef`**: Links tool parameters to live context data for smart constraints
- **`max_call_depth`**: Controls how many rounds of tool calling are allowed per turn

## Next Steps

Now you're ready to build sophisticated AI agents! Try:

- Adding more complex tools (file I/O, API calls, data analysis)
- Creating multi-agent systems that collaborate
- Building domain-specific agents for your use cases
- Exploring advanced context patterns and validation

The declarative nature of PyAgentic makes it easy to iterate and extend your agents as your needs grow.
