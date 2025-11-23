# CHANGELOG

<!-- version list -->

## v2.2.1-a.1 (2025-11-23)

### Bug Fixes

- Updated tool defs to get based on phase and given condition for tools and linked agents
  ([`8280669`](https://github.com/rmikulec/pyAgentic/commit/82806698c70313331719504578513c30e9fcf7d7))


## v2.2.0 (2025-11-23)
## v2.2.0-a.2 (2025-11-23)

### Bug Fixes

- Update agent to run parallel tool calls actually in parallel
  ([`693b321`](https://github.com/rmikulec/pyAgentic/commit/693b321383af0945d16a1961d0da5ebfc23b0b7a))


## v2.2.0-a.1 (2025-11-23)

### Features

- Added new phase machine FSM system
  ([`9046d25`](https://github.com/rmikulec/pyAgentic/commit/9046d258e0e66a4ca32e74eda4428a826fab20c5))


## v2.1.0 (2025-11-23)


## v2.1.0-a.3 (2025-11-23)

### Bug Fixes

- Removed old emitter in replace for .step
  ([`02d004d`](https://github.com/rmikulec/pyAgentic/commit/02d004d4fc54f6ffd11318cc0174ff56faa84fef))


## v2.1.0-a.2 (2025-11-23)


## v2.1.0-a.1 (2025-11-23)

### Bug Fixes

- Updated ToolReponse to respect the specified return type of the tool
  ([`5b87c5c`](https://github.com/rmikulec/pyAgentic/commit/5b87c5c67a1f23ef644e6122363c074590cd4178))

### Features

- Updated `run` to yield responses -> ToolResponse, LLMResponse, AgentResponse
  ([`7499b39`](https://github.com/rmikulec/pyAgentic/commit/7499b390e29b4aca1c9bf740545e428caa54285d))


## v2.0.2 (2025-11-22)


## v2.0.1-a.2 (2025-11-22)

### Bug Fixes

- Added an install D2 step to docs workflow
  ([`11064f4`](https://github.com/rmikulec/pyAgentic/commit/11064f4f7104d5297d871cf7af685df8fd35922c))


## v2.0.1-a.1 (2025-11-22)

### Bug Fixes

- Quick change to force version bump
  ([`c6f92a8`](https://github.com/rmikulec/pyAgentic/commit/c6f92a89c580ca58b926ea67b4d792a5106588e0))


## v2.0.0-b.1 (2025-11-22)

### Refactoring

- Major State Update
  ([`7078e07`](https://github.com/rmikulec/pyAgentic/commit/7078e07d8aa9220f36807f1f17bc29c274e793f3))


## v1.8.0 (2025-11-22)


## v1.8.0-b.1 (2025-11-22)


## v3.0.0 (2025-11-22)


## v2.0.0 (2025-11-22)


## v1.8.0-a.3 (2025-11-22)


## v1.8.0-a.2 (2025-11-22)

### Features

- Agent links are now working with spec
  ([`3f288cb`](https://github.com/rmikulec/pyAgentic/commit/3f288cb0bb9790835403e36afade9e3befbc3f80))


## v1.8.0-a.1 (2025-11-05)

### Bug Fixes

- Changed tool def failed to invalid tool def exception
  ([`a909a64`](https://github.com/rmikulec/pyAgentic/commit/a909a64d3b1a2f623bdf70f653e72b9a04b78d34))

- Made error handling in tool calling better
  ([`75e6d4f`](https://github.com/rmikulec/pyAgentic/commit/75e6d4f3f4e8f258fc3dc59b8feb4b8265c03642))

- Moved agent base code to subfolder for readability
  ([`bf56103`](https://github.com/rmikulec/pyAgentic/commit/bf5610310d1462140c5eb5e9952342cc25fcc97e))

- Moved policy to new submadule
  ([`9cb04f6`](https://github.com/rmikulec/pyAgentic/commit/9cb04f630af5f9e26f18bba5d2368136ecbaf16d))

- Removed depreactated Param class, Tools now can use pydantic BaseModels
  ([`2c6a7b2`](https://github.com/rmikulec/pyAgentic/commit/2c6a7b21885b658affa70e3de473540da58afa46))

### Features

- Massive docstring update
  ([`f2a550f`](https://github.com/rmikulec/pyAgentic/commit/f2a550fb8d5ecaffb746d2e15a0788b71fca8fc9))

- Updated tool to use pydantic BaseModel rather than custom Param
  ([`3846751`](https://github.com/rmikulec/pyAgentic/commit/384675132f8a3a29121edc3ed1e641af3cce6a4b))


## v1.7.0 (2025-09-18)


## v1.7.0-a.5 (2025-09-18)

### Bug Fixes

- Moved agent tracing tests to better spot
  ([`a227a01`](https://github.com/rmikulec/pyAgentic/commit/a227a015a046e62d2da5c44cc09c058654e22451))

### Features

- Added tests for tracing
  ([`0f007f5`](https://github.com/rmikulec/pyAgentic/commit/0f007f5c5357997787fb628191c3e819e317942a))


## v1.7.0-a.4 (2025-09-18)

### Bug Fixes

- Link llm response usage to tracer
  ([`4358595`](https://github.com/rmikulec/pyAgentic/commit/435859551cfe9dafd27b1feb1a4c5c79538e208b))

- Updated agent run test to incorporate usage
  ([`b8ead5b`](https://github.com/rmikulec/pyAgentic/commit/b8ead5ba135cb849b9e84ef469269922ef4b1f75))


## v1.7.0-a.3 (2025-09-18)


## v1.7.0-a.2 (2025-09-18)


## v1.7.0-a.1 (2025-09-18)

### Bug Fixes

- Added recent_message helper to context
  ([`ed79b4e`](https://github.com/rmikulec/pyAgentic/commit/ed79b4e8db0fe8f273568ff03706f6fae2572a86))

- Added tracer to post init and cleaned up code a bit
  ([`7ae8c68`](https://github.com/rmikulec/pyAgentic/commit/7ae8c68550711468d41d79c938f89cbc75386b52))

### Features

- Added new models for tracing
  ([`231cd87`](https://github.com/rmikulec/pyAgentic/commit/231cd8733d0c3147dab0efc0ffaa3bbc62afe740))


## v1.6.2-a.1 (2025-09-18)

### Bug Fixes

- Moved base tracer to tracing submodule
  ([`33a2a8e`](https://github.com/rmikulec/pyAgentic/commit/33a2a8e79fdbd02aa902e205026d20a3c22e844b))


## v1.6.1 (2025-09-18)


## v1.6.1-a.1 (2025-09-18)

### Bug Fixes

- Added anthropic package
  ([`0e65315`](https://github.com/rmikulec/pyAgentic/commit/0e653154ea58a5fed482db5b053abe3ee19b732d))


## v1.6.0 (2025-09-18)


## v1.6.0-a.1 (2025-09-18)

### Bug Fixes

- Accidently overwrote the pyproject toml
  ([`a12ba71`](https://github.com/rmikulec/pyAgentic/commit/a12ba710c8a61ac7db335a7236402a3998c0749e))

- Manually bumping version
  ([`4f82045`](https://github.com/rmikulec/pyAgentic/commit/4f82045311768ad1aa13834496c3795ce3753bed))

- Updated readme for new providers
  ([`ec1d840`](https://github.com/rmikulec/pyAgentic/commit/ec1d84053236f90df3b66f877ec7420d3692b165))

### Features

- Added suppport for more providers, such as anthropic
  ([`851a2b9`](https://github.com/rmikulec/pyAgentic/commit/851a2b9923b963a45e2b05670233d03bbf341e35))


## v1.4.1 (2025-08-12)


## v1.4.1-a.1 (2025-08-12)


## v1.4.0 (2025-08-12)


## v1.4.0-a.1 (2025-08-12)

### Features

- Updated agent to be able to call responses api multipel times in one run call, allowing it to
  string together tools
  ([`f544ac2`](https://github.com/rmikulec/pyAgentic/commit/f544ac28a41ff3119bcd3796acf7e6b5299945b1))


## v1.3.1-a.1 (2025-08-11)

### Bug Fixes

- Updating docs to be more clear and concise
  ([`88c7eca`](https://github.com/rmikulec/pyAgentic/commit/88c7eca85bc79c074c0bed539981c66b2255bea9))

## v1.3.0 (2025-08-11)


## v1.3.0-a.4 (2025-08-11)

### Bug Fixes

- Made metaclass thread safe as well as made class attribute immutable
  ([`4ff5e1a`](https://github.com/rmikulec/pyAgentic/commit/4ff5e1aa709fbeadc313ec83b403593b91e7dd6d))


## v1.3.0-a.3 (2025-08-08)

### Bug Fixes

- Reworked inheritance to follow MRO order
  ([`283b3aa`](https://github.com/rmikulec/pyAgentic/commit/283b3aaf649be9002da1e7d869117c5e096ef3a3))

### Features

- Added agent extension to allow users to declare mixins
  ([`d79e268`](https://github.com/rmikulec/pyAgentic/commit/d79e2689b973c62851243fd10d38077639d3ca4a))

- Added agent inheritance
  ([`a5cad6e`](https://github.com/rmikulec/pyAgentic/commit/a5cad6e61000d379046d2106041f58b0f53671a5))


## v1.3.0-a.2 (2025-08-08)


## v1.3.0-a.1 (2025-08-08)

### Features

- Implemented a rough outline on how agents are linked
  ([`da95d8c`](https://github.com/rmikulec/pyAgentic/commit/da95d8c6ef1439a5c023665ce0d0c0b5d3592527))


## v1.2.1 (2025-08-08)

### Bug Fixes

- Added support for nested params in tool arg compile
  ([`a455b3f`](https://github.com/rmikulec/pyAgentic/commit/a455b3f57156653d4fff5f3dcd2ed336839ae64d))

- Removed unneeded code in context
  ([`f0c1ea8`](https://github.com/rmikulec/pyAgentic/commit/f0c1ea8ff5d1cda4bdd066dd29cc84efaaad6316))


## v1.2.0 (2025-08-07)


## v1.2.0-a.1 (2025-08-07)


## v1.1.0-a.2 (2025-08-07)


## v1.1.0-a.1 (2025-08-07)

### Features

- Adding a getting started page to the docs
  ([`577b794`](https://github.com/rmikulec/pyAgentic/commit/577b794b6e8eff526de235d2adfc7a0b67b9fb10))
## v1.1.0 (2025-08-07)


## v1.0.1-a.1 (2025-08-07)

### Bug Fixes

- Ensuring openai package is up-to-date
  ([`a9653ea`](https://github.com/rmikulec/pyAgentic/commit/a9653eabd5b4fab573855c61add1336e5c11f268))


## v1.0.0 (2025-08-07)


## v1.0.0-a.5 (2025-08-07)


## v1.0.0-a.4 (2025-08-07)


## v1.0.0-a.3 (2025-08-07)


## v1.0.0-a.2 (2025-08-07)

### Bug Fixes

- Testing out auto-release
  ([`0df5016`](https://github.com/rmikulec/pyAgentic/commit/0df5016c347d768a5c2c60e100eecc6f6d8bad57))


## v1.0.0-a.1 (2025-08-07)

### Bug Fixes

- Test
  ([`8704265`](https://github.com/rmikulec/pyAgentic/commit/8704265f525a5c6df856b4d7966a421c0532a400))


## v0.0.0 (2025-08-06)

- Initial Release
