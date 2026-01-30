# Product Guidelines

## Voice and Tone

**Friendly and approachable** - Documentation and UI text should be accessible to musicians who aren't developers. Avoid jargon where possible; when technical terms are necessary, explain them in context.

## Design Principles

1. **Audio quality above all** - Never compromise output quality for speed or simplicity. If a trade-off exists between latency and quality, default to quality unless the user explicitly opts for speed.

2. **Performance first** - Every architectural decision optimizes for low latency and GPU utilization. Prefer GPU-accelerated paths, minimize CPU-GPU data transfers, and batch operations where possible.

3. **State-of-the-art by default** - Always incorporate the most current research and techniques. When implementing a feature, research and use SOTA approaches rather than legacy methods. This includes:
   - Latest model architectures (not just "good enough" ones)
   - Current best practices from recent papers and benchmarks
   - Modern training techniques (loss functions, augmentation, scheduling)
   - Cutting-edge inference optimizations (quantization, distillation, graph optimization)
   - If a newer, better approach exists, prefer it over established-but-outdated methods

## Guidelines

- Error messages should explain what went wrong AND what the user can do about it
- Audio processing failures should never silently degrade quality - fail loudly instead
- Progress feedback for long operations (training, conversion) via real-time WebSocket updates
- Model training should show clear metrics (loss curves, quality estimates) so users know when to stop
- File formats should be clearly documented and standard (WAV, FLAC preferred over proprietary)
