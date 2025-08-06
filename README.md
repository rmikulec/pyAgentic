# PyAgentic

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/pyagentic/pyagentic/workflows/Tests/badge.svg)](https://github.com/pyagentic/pyagentic/actions)

A declarative framework for building AI agents with OpenAI integration. PyAgentic provides a clean, type-safe way to create intelligent agents using Python's metaclass system and modern async patterns.

## ✨ Features

- **Declarative Agent Definition** - Define agents using simple class-based syntax
- **Type Safety** - Full typing support with Pydantic integration
- **Tool Integration** - Easy function decoration for agent capabilities
- **Context Management** - Sophisticated context handling with lifecycle management
- **OpenAI Integration** - Native support for OpenAI's API with automatic schema generation
- **Async Support** - Built-in async/await support for scalable applications
- **Extensible** - Clean architecture for custom tools, context types, and validations

## 🚀 Quick Start

### Installation

```bash
pip install pyagentic
```

### Basic Example

```python
from pyagentic import Agent, tool, ContextItem
from typing import List

class WeatherAgent(Agent):
    """An agent that provides weather information."""
    
    location: str = ContextItem(description="Current location")
    
    @tool
    def get_weather(self, city: str) -> str:
        """Get current weather for a city."""
        # Your weather API logic here
        return f"The weather in {city} is sunny and 75°F"
    
    @tool
    def get_forecast(self, city: str, days: int = 5) -> List[str]:
        """Get weather forecast for multiple days."""
        return [f"Day {i+1}: Partly cloudy" for i in range(days)]

# Create and use the agent
agent = WeatherAgent(location="San Francisco")
response = await agent.run("What's the weather like in New York?")
print(response)
```

## 📚 Documentation

- **[User Guide](docs/user-guide/)** - Complete guide to using PyAgentic
- **[API Reference](docs/api-reference/)** - Detailed API documentation
- **[Examples](examples/)** - Real-world examples and use cases
- **[Contributing](docs/contributor-guide/contributing.md)** - How to contribute to PyAgentic

## 🏗️ Project Structure

```
pyagentic/
├── pyagentic/           # Core framework code
│   ├── _base/           # Internal implementation
│   └── __init__.py      # Public API
├── tests/               # Test suite
│   ├── _base/           # Core tests
│   ├── integration/     # Integration tests
│   └── performance/     # Performance tests
├── examples/            # Example agents
├── templates/           # Agent templates
├── docs/                # Documentation
├── scripts/             # Utility scripts
└── notebooks/           # Jupyter notebooks
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributor-guide/contributing.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/pyagentic/pyagentic.git
cd pyagentic

# Install in development mode
make install-dev

# Run tests
make test

# Format code
make format
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for their excellent API and models
- The Python community for amazing tools and libraries
- All contributors who help make this project better

---

**Made with ❤️ by the PyAgentic Team**
