from typing import Callable

import gymnasium
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ObsType, WrapperObsType


class ObservationPreprocessingWrapper(ObservationWrapper):
    """
    Wrapper that applies a given preprocessing function to the observation.
    """
    def __init__(self,
                 env: gymnasium.Env,
                 obs_space: gymnasium.Space,
                 preprocess_fn: Callable):
        """
        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        preprocess_fn : Callable
            The function to apply to the observation.
        """
        super().__init__(env)
        self._preprocess_fn = preprocess_fn
        self.observation_space = obs_space

    def observation(self, observation: ObsType) -> WrapperObsType:
        """
        Apply the preprocessing function to the observation.

        Parameters
        ----------
        observation : object
            The original observation to preprocess.

        Returns
        -------
        object
            The preprocessed observation.
        """
        obs = self._preprocess_fn(observation)
        return obs