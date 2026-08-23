
# CUBE Environment Constants
env_rule = {
    "llm_model_name": "gpt-4o",
    "num_agents": 3,
    "env_setting": "default",
}

# CUBE only has 5 primitive actions
ACTION_NAME_TO_VALUE = {
    "noop": 0,
    "stay": 0,
    "move_up": 1,
    "move_down": 2,
    "move_left": 3,
    "move_right": 4,
    # Aliases for convenience
    "up": 1,
    "down": 2, 
    "left": 3,
    "right": 4,
}

# For CUBE, we only have move and wait actions
ACTION_SCHEMA = {
    "move": [
        {"type": "direction", "field": "direction"},
        {"type": "int", "field": "num_steps"},
    ],
    "wait": [
        {"type": "int", "field": "num_steps"},
    ],
}
