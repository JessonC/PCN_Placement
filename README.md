# PCN Placement

This repository contains utilities for experimenting with placement algorithms.

## New RL Example

The `layout_rl` directory demonstrates a minimal reinforcement learning setup for
component placement. A simple grid based environment is provided together with a
small PPO implementation. To run a quick training session execute:

```bash
python -m layout_rl.train
```

The environment and agent are intentionally lightweight so they can run without
extra dependencies.
