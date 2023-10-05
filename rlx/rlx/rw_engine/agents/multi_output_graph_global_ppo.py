import os
import time
import datetime
from typing import Dict
import numpy as np

from rlx.utils.common import get_logger
from rlx.rw_engine.agents.gnn import GATNetwork, GATNetwork_with_global, CategoricalMasked, layer_init, GlobalLayer, PoolingLayer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import torch_geometric as pyg

MatchDict = Dict[int, tuple[int, int]]  # pattern_id -> bool, edge_id/node_id

logger = get_logger(__name__)


class MultiOutputGraphGlobalPPO(nn.Module):

    def __init__(self,
                 nvec,
                 n_node_features: int,
                 n_edge_features: int,
                 num_head: int,
                 n_layers: int,
                 hidden_size: int,
                 vgat: int,
                 use_dropout=False,
                 device=torch.device("cpu")):
        super().__init__()
        logger.info(f"nvec: {nvec}")
        logger.info(f"actor_n_action: {nvec}")
        logger.info(f"critic_n_action: 1")
        logger.info(f"use_dropout: {use_dropout}")
        assert vgat == 1 or vgat == 2, f"vgat must be 1 or 2, got {vgat}"
        self.nvec = nvec.tolist()  # list[int]
        self.device = device

        self.critic = GATNetwork_with_global(
            num_node_features=n_node_features,
            num_edge_features=n_edge_features,
            n_actions=1,
            n_layers=n_layers,
            hidden_size=hidden_size,
            num_head=num_head,
            vgat=vgat,
            dropout=(0.3 if use_dropout else 0.0),
        )

        self.gnn = GATNetwork(
            num_node_features=n_node_features,
            num_edge_features=n_edge_features,
            n_actions=hidden_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            num_head=num_head,
            vgat=vgat,
            dropout=(0.3 if use_dropout else 0.0),
        )
        self.actor1 = PoolingLayer(
            hidden_size,
            self.nvec[0],
            out_std=0.001,
        )
        self.actor2 = GlobalLayer(
            hidden_size,
            self.nvec[1],
            out_std=0.001,
        )

    def get_value(self, x):
        vf, _ = self.critic(x)
        return vf

    def get_action_and_value(self,
                             x: pyg.data.Batch,
                             invalid_rule_mask: torch.Tensor,
                             invalid_loc_mask: torch.Tensor,
                             action=None):
        hidden, _ = self.gnn(x)
        vf, _ = self.critic(x)

        num_graphs = x.num_graphs
        batch = x.batch
        # print(f"numGraph: {num_graphs}")

        logits_1 = self.actor1(hidden, batch)
        c1 = CategoricalMasked(logits=logits_1,
                               mask=invalid_rule_mask,
                               device=self.device)
        if action is None:
            # inference
            a1 = c1.sample()

            # shape: [batch_size, max_loc]
            loc_mask = invalid_loc_mask[
                torch.arange(invalid_loc_mask.shape[0]), a1]
            # print("loc_mask", loc_mask.shape)

            # a1 is used as graph global features
            logits_2 = self.actor2(
                hidden,
                x.edge_index,
                None,
                a1.reshape(num_graphs, 1),
                batch,
            )
            c2 = CategoricalMasked(logits=logits_2,
                                   mask=loc_mask,
                                   device=self.device)
            a2 = c2.sample()
            # print(a1.shape, a2.shape, logits_2.shape)
            action = torch.stack([a1, a2], dim=0)  # transpose later
            #print(a1, a2)
            #print(action)
        else:
            # training
            a1 = action[:, 0]
            loc_mask = invalid_loc_mask[
                torch.arange(invalid_loc_mask.shape[0]), a1]

            logits_2 = self.actor2(
                hidden,
                x.edge_index,
                None,
                a1.reshape(num_graphs, 1),
                batch,
            )
            c2 = CategoricalMasked(logits=logits_2,
                                   mask=loc_mask,
                                   device=self.device)
            action = action.T

        log_prob = torch.stack(
            [c.log_prob(a) for a, c in zip(action, [c1, c2])])
        # print([c.log_prob(a) for a, c in zip(action, [c1, c2])])
        # print(log_prob)

        entropy = torch.stack([c.entropy() for c in [c1, c2]])
        return action.T, log_prob.sum(0), entropy.sum(0), vf


def env_loop(envs, config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(config.gpu) else "cpu")
    logger.info(f"device: {device}")
    # ===== env =====
    state, _ = envs.reset(seed=config.seed)

    # ===== log =====
    log = bool(config.l)
    if log:
        t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file = config.fn.split("/")[-1] if config.fn is not None else config.dir
        run_name = f"rlx_{config.env_id}__{config.agent}__{file}"
        if config.annotation is not None:
            run_name += f"__{config.annotation}"
        run_name += f"__{t}"
        save_path = f"{config.default_out_path}/runs/{run_name}"
        writer = SummaryWriter(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        config.append_flags_into_file(save_path + "/flags.txt")

        logger.info(f"[ENV_LOOP]save path: {save_path}")

    # ===== agent =====
    agent = MultiOutputGraphGlobalPPO(
        nvec=envs.single_action_space.nvec,
        n_node_features=envs.single_observation_space.node_space.shape[0],
        n_edge_features=envs.single_observation_space.edge_space.shape[0],
        num_head=config.num_head,
        n_layers=config.n_layers,
        hidden_size=config.hidden_size,
        vgat=config.vgat,
        use_dropout=bool(config.use_dropout),
        device=device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # ===== START GAME =====
    # ALGO Logic: Storage setup
    # gh512
    # obs = torch.zeros((config.num_steps, config.num_envs) +
    #                   envs.single_observation_space.shape).to(device)
    # obs = []  # collect graphs
    actions = torch.zeros(
        (config.num_steps, config.num_envs) + envs.single_action_space.shape,
        dtype=torch.long).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    invalid_rule_masks = torch.zeros(
        (config.num_steps, config.num_envs, envs.single_action_space.nvec[0]),
        dtype=torch.long).to(device)
    invalid_loc_masks = torch.zeros(([
        config.num_steps,
        config.num_envs,
    ] + envs.single_action_space.nvec.tolist()),
                                    dtype=torch.long).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # gh512
    # next_obs = torch.Tensor(envs.reset()).to(device)
    next_obs = pyg.data.Batch.from_data_list([i[0] for i in state]).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    invalid_rule_mask = torch.cat([i[2] for i in state]).reshape(
        (config.num_envs, -1)).to(device)
    invalid_loc_mask = torch.cat([
        i[3] for i in state
    ]).reshape([config.num_envs] +
               envs.single_action_space.nvec.tolist()).to(device)

    # batch size
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_mini_batch)
    num_updates = config.total_timesteps // batch_size
    logger.info(f"[ENV_LOOP] minibatch size: {minibatch_size}")

    # resolve int config
    anneal_lr = bool(config.anneal_lr)
    gae = bool(config.gae)
    norm_adv = bool(config.norm_adv)
    clip_vloss = bool(config.clip_vloss)

    # utility
    save_freq = (num_updates + 1) // 10
    best_episodic_return = -float("inf")

    # ===== RUN =====
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.lr
            optimizer.param_groups[0]["lr"] = lrnow

        # ==== rollouts ====
        # gh512, reset;
        obs = []  # list[pyg.Data]; reset each update
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            # gh512
            # obs[step] = next_obs
            obs.append(next_obs)
            dones[step] = next_done

            # print(invalid_rule_masks.shape)
            # print(invalid_rule_masks[step].shape)
            # print(invalid_rule_mask.shape)
            invalid_rule_masks[step] = invalid_rule_mask
            invalid_loc_masks[step] = invalid_loc_mask

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs,
                    invalid_rule_mask=invalid_rule_mask,
                    invalid_loc_mask=invalid_loc_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # gh512
            a = [tuple(i) for i in action.cpu()]
            next_obs, reward, terminated, truncated, infos = envs.step(a)
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # gh512
            # next_obs, next_done = torch.Tensor(next_obs).to(
            #     device), torch.Tensor(done).to(device)
            next_done = torch.Tensor(done).to(device)
            invalid_rule_mask = torch.cat([i[2] for i in next_obs]).reshape(
                (config.num_envs, -1)).to(device)
            invalid_loc_mask = torch.cat([
                i[3] for i in next_obs
            ]).reshape([config.num_envs] +
                       envs.single_action_space.nvec.tolist()).to(device)

            next_obs = pyg.data.Batch.from_data_list([i[0] for i in next_obs
                                                      ]).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                # this is recorded by RecordEpisodeStatistics wrapper
                logger.info(
                    f"[ENV_LOOP] global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}, episodic_time={info['episode']['t']}"
                )
                if log:
                    writer.add_scalar("charts/episodic_return",
                                      info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length",
                                      info["episode"]["l"], global_step)

        # ==== after rollouts, updates ====
        # bootstrap value if not done
        # print(f"[gae] {update}", end="\t")
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # _, _, _, next_value = agent.get_action_and_value(
            #     next_obs,
            #     invalid_rule_mask=invalid_rule_mask,
            #     invalid_loc_mask=invalid_loc_mask)
            # next_value = next_value.reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[
                        t] + config.gamma * nextvalues * nextnonterminal - values[
                            t]
                    advantages[
                        t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[
                        t] + config.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        # gh512
        # b_obs = obs.reshape((-1, ) + envs.single_observation_space.shape)
        # NOTE: `unbatch` graphs
        b_obs = []
        for batch_graph in obs:
            per_step_graphs = pyg.data.Batch.to_data_list(batch_graph)
            for per_step_per_env_graph in per_step_graphs:
                b_obs.append(per_step_per_env_graph)

        # those buffers are in device
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, ) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_rule_masks = invalid_rule_masks.reshape(
            config.num_steps * config.num_envs, -1)
        b_invalid_loc_masks = invalid_loc_masks.reshape(
            [config.num_steps * config.num_envs] +
            envs.single_action_space.nvec.tolist())

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        # print(f"[update] {update}")
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # print("indx")
                # print(batch_size)
                # print(mb_inds)
                # print(len(b_obs))
                # print(b_obs[0])
                # tmp = pyg.data.Batch.to_data_list(b_obs[0])
                # print("tmp")
                # print(len(tmp))

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    # gh512 (re-batch here)
                    # b_obs[mb_inds],
                    pyg.data.Batch.from_data_list([b_obs[i] for i in mb_inds]
                                                  ).to(device),
                    invalid_rule_mask=b_invalid_rule_masks[mb_inds],
                    invalid_loc_mask=b_invalid_loc_masks[mb_inds],
                    action=b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs()
                                   > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        # ==== after updates, log ====
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                             y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        t = int(global_step / (time.time() - start_time))
        logger.info(f"[ENV_LOOP] Update: {update}/{num_updates} SPS: {t}")
        # logger.info(f"[ENV_LOOP] episodic_return: {episodic_return}")
        if log:
            writer.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(),
                              global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(),
                              global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(),
                              global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(),
                              global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs),
                              global_step)
            writer.add_scalar("losses/explained_variance", explained_var,
                              global_step)
            writer.add_scalar("charts/SPS", t, global_step)

        if log and update % save_freq == 0:
            torch.save(agent.state_dict(),
                       f"{save_path}/agent-{global_step}.pt")

    # ===== STOP =====
    envs.close()
    if log:
        # save
        torch.save(agent.state_dict(), f"{save_path}/agent-final.pt")
        writer.close()


def inference(env, config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(config.gpu) else "cpu")
    logger.info(f"device: {device}")
    # ===== env =====
    state, _ = env.reset(seed=config.seed)

    # ===== agent =====
    assert config.weights_path is not None, "weights_path must be set"
    agent_id = config.agent_id if config.agent_id is not None else "agent-final"
    fn = os.path.join(f"{config.default_out_path}/runs/", config.weights_path,
                      f"{agent_id}.pt")
    agent = MultiOutputGraphGlobalPPO(
        nvec=env.action_space.nvec,
        n_node_features=env.observation_space.node_space.shape[0],
        n_edge_features=env.observation_space.edge_space.shape[0],
        num_head=config.num_head,
        n_layers=config.n_layers,
        hidden_size=config.hidden_size,
        vgat=config.vgat,
        use_dropout=bool(config.use_dropout),
        device=device)
    agent_state_dict = torch.load(fn, map_location=device)
    agent.load_state_dict(agent_state_dict)
    agent.to(device)

    next_obs = state[0].to(device)
    invalid_rule_mask = state[2].reshape(1, -1).to(device)
    invalid_loc_mask = state[3].reshape(
        [config.num_envs] + env.action_space.nvec.tolist()).to(device)

    # ==== rollouts ====
    cnt = 0
    done = False
    while not done:
        cnt += 1
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                next_obs,
                invalid_rule_mask=invalid_rule_mask,
                invalid_loc_mask=invalid_loc_mask)

        # TRY NOT TO MODIFY: execute the game and log data.
        action = action[0].cpu()
        a = [action[0], action[1]]
        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = np.logical_or(terminated, truncated)
        invalid_rule_mask = next_obs[2].reshape(1, -1).to(device)
        invalid_loc_mask = next_obs[3].reshape(
            [config.num_envs] + env.action_space.nvec.tolist()).to(device)
        next_obs = next_obs[0].to(device)

        logger.info(f"iter {cnt}; reward: {reward}")

    print(terminated, truncated)
    #print(info)
