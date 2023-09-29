from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.env import make_env, InferenceVecEnv

# Agent table
from rlx.rw_engine.agents.multi_output_ppo import env_loop as multi_output_ppo
from rlx.rw_engine.agents.multi_output_ppo import inference as multi_output_ppo_inference

from rlx.rw_engine.agents.multi_output_graph_global_ppo import env_loop as multi_output_graph_global_ppo_env_loop
from rlx.rw_engine.agents.multi_output_graph_global_ppo import inference as multi_output_graph_global_ppo_inference

from rlx.rw_engine.agents.multi_output_max_ppo import env_loop as multi_output_max_ppo_env_loop
from rlx.rw_engine.agents.multi_output_max_ppo import inference as multi_output_max_ppo_inference

from rlx.rw_engine.agents.ppo import env_loop as ppo_env_loop
from rlx.rw_engine.agents.ppo import inference as ppo_inference

import gymnasium as gym


class RewriteEngine:
    def __init__(
        self,
        graphs,
        rewrite_rules,
        callback_reward_function,
        config,
    ):
        self.graphs = graphs
        self.callback_reward_function = callback_reward_function
        for rule in rewrite_rules:
            rule.initialise()
        self.rewrite_rules = rewrite_rules
        self.config = config

    def run(self):
        """run inference to transform the target graphs"""
        # ===== Internal IR =====
        parsers = [Parser([g]) for g in self.graphs]

        # ===== env ===== (each env has a parser)
        envs = InferenceVecEnv(
            [
                make_env(env_id=self.config.env_id,
                         parser=p,
                         callback_reward_function=lambda x, y, z, _: 0,
                         rewrite_rules=self.rewrite_rules,
                         seed=self.config.seed,
                         config=self.config) for p in parsers
            ],
            copy=False,  # no need to deepcopy at step
        )

        # ===== dispatch to agent inference =====
        if self.config.agent == "multi_output_ppo":
            inference = multi_output_ppo_inference
        elif self.config.agent == "multi_output_graph_global_ppo":
            inference = multi_output_graph_global_ppo_inference
        elif self.config.agent == "multi_output_max_ppo":
            inference = multi_output_max_ppo_inference
        elif self.config.agent == "ppo":
            inference = ppo_inference
        else:
            raise RuntimeError(f"fail to dispatch {self.config.agent}")

        opt_time = inference(envs, self.config)
        self.envs = envs
        return opt_time

    def train(self):
        """train the RL given a batch of training graphs"""
        # ===== Internal IR =====
        parser = Parser(self.graphs)

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
            envs = gym.vector.SyncVectorEnv(
                [
                    make_env(
                        env_id=self.config.env_id,
                        parser=parser,
                        callback_reward_function=self.callback_reward_function,
                        rewrite_rules=self.rewrite_rules,
                        seed=self.config.seed + i,
                        config=self.config)
                    for i in range(self.config.num_envs)
                ],
                copy=False,  # no need to deepcopy at step
            )

        # ===== dispatch to agent training loop =====
        if self.config.agent == "multi_output_ppo":
            multi_output_ppo(envs, self.config)
        elif self.config.agent == "multi_output_graph_global_ppo":
            multi_output_graph_global_ppo_env_loop(envs, self.config)
        elif self.config.agent == "multi_output_max_ppo":
            multi_output_max_ppo_env_loop(envs, self.config)
        elif self.config.agent == "ppo":
            ppo_env_loop(envs, self.config)
        else:
            raise RuntimeError(f"fail to dispatch {self.config.agent}")
