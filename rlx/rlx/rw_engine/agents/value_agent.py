import time
import datetime
from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import torch_geometric as pyg


class GraphValue(nn.Module):
    def __init__(self,
                 n_actions: int,
                 n_node_features: int,
                 n_edge_features: int,
                 weights_path=None,
                 use_dropout=False,
                 use_edge_attr=True,
                 device=torch.device("cpu")):
        super().__init__()
        print()
        print("[GraphValue] init::")
        print("use_edge_attr: ", use_edge_attr, " use_dropout: ", use_dropout,
              " weights_path: ", weights_path)
        print()
        self.device = device
        self.critic = GATNetwork_with_global(
            num_node_features=n_node_features,
            n_actions=1,
            n_layers=3,
            hidden_size=64,
            dropout=(0.3 if use_dropout else 0.0),
            use_edge_attr=use_edge_attr,
            edge_dim=n_edge_features,
            out_std=1.)

        # GATNetwork, GATNetwork_v2
        self.actor = GATNetwork_v2(
            num_node_features=n_node_features,
            n_actions=n_actions,
            n_layers=3,
            hidden_size=64,
            dropout=(0.3 if use_dropout else 0.0),
            use_edge_attr=use_edge_attr,
            # no need to make action similar
            # out_std=0.001,
            edge_dim=n_edge_features)

        # load from pre-trained
        if weights_path is not None:
            self.actor.load_state_dict(torch.load(weights_path))
            keys_vin = torch.load(weights_path)
            current_model = self.critic.state_dict()
            new_state_dict = {
                k:
                v if v.size() == current_model[k].size() else current_model[k]
                for k, v in zip(current_model.keys(), keys_vin.values())
            }
            self.critic.load_state_dict(new_state_dict, strict=False)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, invalid_action_mask, action=None):
        logits = self.actor(x)

        # print(f"logits:: {logits.shape} {invalid_action_mask.shape}")
        if invalid_action_mask is not None:
            probs = CategoricalMasked(logits=logits,
                                      mask=invalid_action_mask,
                                      device=self.device)
        else:
            probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GATNetwork_v2(nn.Module):
    def __init__(
            self,
            num_node_features: int,
            n_actions: int,  # fix action space = num of nodes (Eclass + Enode)
            n_layers: int = 3,
            hidden_size: int = 128,
            out_std=np.sqrt(2),
            dropout=0.0,
            use_edge_attr=True,
            add_self_loops=False,
            edge_dim=None):
        super().__init__()
        self.n_actions = n_actions
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            assert (edge_dim is not None), "use edge attr but edge_dim is None"
        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True,
                       edge_dim=(edge_dim if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                # convert node-level feature to scalar, representing its logit
                layer_init(nn.Linear(hidden_size, 1), std=out_std))
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                nn.Linear(hidden_size, 1))

        # a final layer to transform those scalar node feat
        self.ff = nn.Sequential(nn.Linear(self.n_actions, self.n_actions),
                                nn.Tanh(),
                                nn.Linear(self.n_actions, self.n_actions))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=(data.edge_attr if self.use_edge_attr else None))
        # print("[GNN] ", x.shape)
        # head will compress node feat from hidden_size -> 1
        x = self.head(x).reshape(-1, self.n_actions)
        # ff will weight nodes differently
        x = self.ff(x)
        # print("[GNN] ", x.shape)
        return x


class GATNetwork_with_global(nn.Module):
    """A Graph Attentional Network (GAT)
    uses global features as vf

    GAT as in
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
    """
    def __init__(
            self,
            num_node_features: int,
            n_actions: int,  # fix action space = 1, embed the graph as vf
            n_layers: int = 3,
            hidden_size: int = 128,
            out_std=np.sqrt(2),
            dropout=0.0,
            use_edge_attr=True,
            add_self_loops=False,
            edge_dim=None):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            assert (edge_dim is not None), "use edge attr but edge_dim is None"
        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True,
                       edge_dim=(edge_dim if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                layer_init(nn.Linear(hidden_size, n_actions), std=out_std))
        else:
            self.head = nn.Sequential(nn.Dropout(p=dropout),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_size, n_actions))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=(data.edge_attr if self.use_edge_attr else None))
        # print("[GNN global] ", x.shape)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        # print("[GNN global] ", x.shape)
        x = self.head(x)
        # print("[GNN global] ", x.shape)
        return x


def env_loop(envs, config):
    """
    The env_loop is coupled with specific algorithm (e.g. PPO <=> on-policy),
        so each agent should implement their env_loop
    """
    # ===== env =====
    state = envs.reset()

    # ===== log =====
    log = bool(config.l)
    viz = bool(config.viz)
    if log:
        run_name = f"{config.env_id}__{config.lang}__{fn}"
        t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name += f"__{config.seed}__{t}"
        save_path = f"{config.default_out_path}/runs/{run_name}"
        print(f"save path: {save_path}")
        writer = SummaryWriter(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        config.append_flags_into_file(save_path + "/hyperparams.txt")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" %
    #     ("\n".join([f"|{key}|{value}|"
    #     for key, value in vars(config).items()])),
    # )

    # ===== agent =====
    agent = GraphValue(device=device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # ===== START GAME =====
    # ALGO Logic: Storage setup
    # gh512
    # obs = torch.zeros((config.num_steps, config.num_envs) +
    #                   envs.single_observation_space.shape).to(device)
    # obs = []  # collect graphs
    actions = torch.zeros((config.num_steps, config.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    # gh512
    invalid_masks = torch.zeros((config.num_steps, config.num_envs,
                                 envs.single_action_space.n)).to(device)
    # print(invalid_masks.shape)
    # print(envs.single_action_space.shape)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # gh512
    # next_obs = torch.Tensor(envs.reset()).to(device)
    next_obs = pyg.data.Batch.from_data_list([i[0] for i in state]).to(device)
    invalid_mask = [i[1] for i in state]  # list[tensor]
    invalid_mask = torch.cat(invalid_mask).reshape(
        (config.num_envs, -1)).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    # batch size
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_mini_batch)
    num_updates = config.total_timesteps // batch_size
    print(f"[GraphValue] minibatch size: {minibatch_size}")

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

        obs = []  # gh512, reset; list[pyg.Data]; reset each update
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            # gh512
            # obs[step] = next_obs
            obs.append(next_obs)
            dones[step] = next_done
            # print(invalid_mask.shape)
            # print(invalid_masks.shape)
            invalid_masks[step] = invalid_mask

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, invalid_action_mask=invalid_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # gh512
            a = [int(i) for i in action.cpu()]
            # next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs, reward, done, info = envs.step(a)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # gh512
            # next_obs, next_done = torch.Tensor(next_obs).to(
            #     device), torch.Tensor(done).to(device)
            next_done = torch.Tensor(done).to(device)
            invalid_mask = [i[1] for i in next_obs]  # list[tensor]
            invalid_mask = torch.cat(invalid_mask).reshape(
                (config.num_envs, -1)).to(device)
            next_obs = pyg.data.Batch.from_data_list([i[0] for i in next_obs
                                                      ]).to(device)

            # info:: list[dict]
            # print(info)
            # count how many envs are done
            cnt = 0
            for item in info:
                if "episode" in item.keys():
                    cnt += 1
            idx = 0
            for item in info:
                # the `episode` is added by RecordEpisodeStatistics
                if "episode" in item.keys():
                    # print on first non-truncated episode
                    # if not item["TimeLimit.truncated"]:
                    #     print(item)
                    #     # once figure out info's structure
                    #     raise

                    # print("=")
                    # print(item)
                    # print("=")
                    # truncated = item["TimeLimit.truncated"]
                    episodic_return = item["episode"]["r"]
                    episodic_length = item["episode"]["l"]

                    improved = False
                    if episodic_return > best_episodic_return:
                        improved = True
                        best_episodic_return = episodic_return

                    # log
                    if log:
                        writer.add_scalar("charts/episodic_return",
                                          item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                          item["episode"]["l"], global_step)

                    # need gym==0.23.0
                    if log and improved and viz:
                        if async_env:
                            # if use async_env + viz, it is likely to crash
                            envs.call("viz_ast",
                                      save_path + f"/ast-{global_step}")
                        else:
                            envs.envs[idx].viz_ast(save_path +
                                                   f"/ast-{global_step}")

                    print(f"global_step={global_step}", end=", ")
                    print(f"done {cnt}", end=", ")
                    print(f"episodic_return={episodic_return:.2f}", end=", ")
                    print(f"episodic_length={episodic_length}")
                    break
                idx += 1

        # bootstrap value if not done
        # print(f"[gae] {update}", end="\t")
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
            tmp = pyg.data.Batch.to_data_list(batch_graph)
            for j in tmp:
                b_obs.append(j)
        # invalid_masks flatten into per step per env
        b_invalid_masks = invalid_masks.reshape(
            config.num_steps * config.num_envs, -1)
        # print(invalid_masks.shape)
        # print(b_invalid_masks.shape)

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, ) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

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
                    invalid_action_mask=b_invalid_masks[mb_inds],
                    action=b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   config.clip_coef).float().mean().item()]

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

        # after update
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                             y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"Update: {update}/{num_updates} SPS:",
              int(global_step / (time.time() - start_time)),
              end=" - ")
        print("episodic_return: ", episodic_return)

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
            writer.add_scalar("charts/SPS",
                              int(global_step / (time.time() - start_time)),
                              global_step)

        if log and update % save_freq == 0:
            torch.save(agent.state_dict(), f"{save_path}/agent-{global_step}")

    # ===== STOP =====
    envs.close()
    if log:
        # save
        torch.save(agent.state_dict(), f"{save_path}/agent-final")
        writer.close()
