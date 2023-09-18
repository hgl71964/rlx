import gymnasium as gym
from gymnasium.envs.registration import register

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
