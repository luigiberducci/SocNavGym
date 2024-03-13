from typing import Callable

import gymnasium
from gymnasium import ActionWrapper


class ActionProcessingWrapper(ActionWrapper):
    def __init__(self, env, action_space: gymnasium.Space, process_fn: Callable):
        super().__init__(env)
        self._process_fn = process_fn
        self.action_space = action_space

    def action(self, action):
        return self._process_fn(action)