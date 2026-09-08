# COOP²-Repair core

Environment-independent part of COOP²-Repair, shared by `cube/` and `ma_crafter/`:

| Module | Responsibility |
| --- | --- |
| `core.py` | Plan views, task specs, constraint results, trace logger, parallel-env adapter base |
| `prediction.py` | Heuristic plan-timeline predictor used by the adapters |
| `evaluator.py` | Pre-execution constraint evaluation over committed plans |
| `repair_controller.py` | Triggers repair, opens the repair channel, applies the cooldown |
| `message_protocol.py` | Repair-message type constants |

This directory is not a package on its own. Each environment's `coop2_repair/__init__.py` adds it to
the package search path, so `coop2_repair.core` and friends resolve to these files from either
environment, and the environment packages contain only their adapters. Edit the shared logic here
once; never copy these files back into an environment.
