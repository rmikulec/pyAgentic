# Architecture Diagrams

This directory contains the source [D2](https://d2lang.com/) files for PyAgentic's architecture diagrams.

## Files

- `declaration.d2` - Shows the declaration phase where the metaclass processes agent class definitions
- `instantiation.d2` - Shows the instantiation phase where agent instances are created
- `runtime.d2` - Shows the runtime execution phase of the agentic loop

## Building Diagrams

The diagrams are automatically compiled to SVG when building or deploying the documentation:

```bash
# Compile all diagrams
uv run task compile-diagrams

# Build docs (automatically compiles diagrams first)
uv run task build-docs

# Serve docs locally (automatically compiles diagrams first)
uv run task serve-docs

# Deploy docs (automatically compiles diagrams first)
uv run task deploy-docs
```

## Layout Engine

The `.d2` files specify `tala` as the layout engine for optimal results, but the build process uses `elk` by default since `tala` requires a separate installation.

To use `tala` for better layouts:

1. Install tala: https://d2lang.com/tour/tala
2. Compile manually without the `--layout` flag:
   ```bash
   d2 declaration.d2 ../declaration.svg
   d2 instantiation.d2 ../instantiation.svg
   d2 runtime.d2 ../runtime.svg
   ```

## Editing Diagrams

1. Edit the `.d2` files in this directory
2. Run `uv run task compile-diagrams` to regenerate the SVGs
3. Preview by serving the docs: `uv run task serve-docs`

For D2 syntax and features, see: https://d2lang.com/tour/intro
