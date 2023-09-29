from copy import deepcopy

import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.vector.utils import concatenate

from rlx.rw_engine.parser import Parser
from rlx.utils.common import get_logger

logger = get_logger(__name__)

register(
    id="env_multi-v0",
    entry_point="rlx.rw_engine.environment.env_multi:Env",
)

register(
    id="env_single-v0",
    entry_point="rlx.rw_engine.environment.env_single:Env",
)


def make_env(
    env_id,
    parser: Parser,
    callback_reward_function: callable,
    rewrite_rules: list,
    seed: int,
    config,
):
    def thunk():
        env = gym.make(env_id,
                       parser=parser,
                       reward_func=callback_reward_function,
                       rewrite_rules=rewrite_rules,
                       max_loc=config.max_loc)

        # utility wrapper
        # env = gym.wrappers.NormalizeReward(env)  # this influences learning significantly
        if config.h is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.h)
        if bool(config.normalize_reward):
            env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # seed env
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    return thunk


class InferenceVecEnv(gym.vector.SyncVectorEnv):
    """Vectorized environment's step will automatically reset
    At inference time, we don't want that
    """
    def step_wait(self):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            # if done; just sample dummy
            if self._terminateds[i] or self._truncateds[i]:
                observation = env.unwrapped.build_dummy()
                info = {}
            else:
                (
                    observation,
                    self._rewards[i],
                    self._terminateds[i],
                    self._truncateds[i],
                    info,
                ) = env.step(action)

            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(self.single_observation_space,
                                        observations, self.observations)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )
