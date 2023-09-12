from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.env import make_env

# Agent table
from rlx.rw_engine.agents.multi_output_ppo import env_loop as multi_output_ppo
from rlx.rw_engine.agents.multi_output_ppo import inference as multi_output_ppo_inference

import gymnasium as gym


class RewriteEngine:

    def __init__(self, graph, rewrite_rules, callback_reward_function, config):
        self.graph = graph
        self.callback_reward_function = callback_reward_function
        for rule in rewrite_rules:
            rule.initialise()
        self.rewrite_rules = rewrite_rules
        self.config = config

    def run(self):
        """run inference to transform the target graph"""
        # ===== Internal IR =====
        parser = Parser(self.graph)

        # ===== env =====
        env = make_env(env_id=self.config.env_id,
                       parser=parser,
                       callback_reward_function=self.callback_reward_function,
                       rewrite_rules=self.rewrite_rules,
                       seed=self.config.seed,
                       config=self.config)()
        # ===== dispatch to agent training loop =====
        if self.config.agent == "multi_output_ppo":
            multi_output_ppo_inference(env, self.config)
        else:
            raise RuntimeError(f"fail to dispatch {self.config.agent}")

        return env.edges

    def train(self):
        """train the RL given a batch of training graphs"""
        # ===== Internal IR =====
        parser = Parser(self.graph)

        # ===== env ===== (SyncVectorEnv, AsyncVectorEnv)
        async_env = bool(self.config.a)
        if async_env:
            envs = gym.vector.AsyncVectorEnv([
                make_env(
                    env_id=self.config.env_id,
                    parser=parser,
                    callback_reward_function=self.callback_reward_function,
                    rewrite_rules=self.rewrite_rules,
                    seed=self.config.seed + i,
                    config=self.config) for i in range(self.config.num_envs)
            ],
                                             shared_memory=False,
                                             copy=False)
        else:
            envs = gym.vector.SyncVectorEnv([
                make_env(
                    env_id=self.config.env_id,
                    parser=parser,
                    callback_reward_function=self.callback_reward_function,
                    rewrite_rules=self.rewrite_rules,
                    seed=self.config.seed + i,
                    config=self.config) for i in range(self.config.num_envs)
            ],
                                            copy=True)

        # ===== dispatch to agent training loop =====
        if self.config.agent == "multi_output_ppo":
            multi_output_ppo(envs, self.config)
        else:
            raise RuntimeError(f"fail to dispatch {self.config.agent}")
