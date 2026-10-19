"""Pinned official GAC networks and learner update, with isolated RNG/checkpoints.

Adapted from Lin-Zhihao98/GAC at 374ae2c9ef284a6ef715a67c1fdce864ebad6f44.
Copyright (c) 2026 Zhihao Lin, MIT; see docs/reference/gac-upstream-LICENSE.txt.

The actor controls the full action vector. This is the official-code recipe,
not a grouped XYZ direction/speed policy or an RLPD actor substitution.
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = '374ae2c9ef284a6ef715a67c1fdce864ebad6f44'


@dataclass(frozen=True)
class GACReferenceConfig:
    gamma: float = .99
    tau: float = .005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = .0003
    q_lr: float = .001
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 1.
    kappa_init: float = 2.
    kappa_min: float = 1.
    kappa_max: float = 20.
    kappa_reg_weight: float = 1.
    radius: float = 1.

    def validate(self):
        if (self.batch_size < 2 or self.learning_starts < 0 or self.policy_frequency < 1
                or self.target_network_frequency < 1
                or any(type(getattr(self,key)) is not int for key in ('batch_size','learning_starts','policy_frequency','target_network_frequency'))):
            raise ValueError('GAC requires batch>=2 and valid integer update intervals.')
        values=[getattr(self,key) for key in ('gamma','tau','policy_lr','q_lr','alpha','kappa_init','kappa_min','kappa_max','kappa_reg_weight','radius')]
        if (not np.isfinite(values).all() or not 0 <= self.gamma <= 1 or not 0 <= self.tau <= 1
                or min(self.policy_lr,self.q_lr,self.kappa_init,self.radius) <= 0
                or min(self.alpha,self.kappa_reg_weight) < 0 or not 0 < self.kappa_min < self.kappa_max):
            raise ValueError('Invalid GAC learner values.')


class SoftQNetwork(nn.Module):

    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GACActor(nn.Module):

    def __init__(self, env, args):
        super().__init__()
        self.args = args
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.action_dim = action_dim
        self.register_buffer('action_scale', torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32))
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, action_dim)
        self.kappa_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.mu_head.weight, gain=0.1)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.constant_(self.kappa_head[-1].bias, np.log(args.kappa_init))
        self.radius = args.radius

    def forward(self, x):
        shared_features = self.backbone(x)
        mu_raw = self.mu_head(shared_features)
        mu = F.normalize(mu_raw, p=2, dim=-1)
        kappa_raw = self.kappa_head(shared_features)

        def soft_clamp(x, min_val, max_val):
            return min_val + (max_val - min_val) * torch.sigmoid(x / 5.0)
        kappa_raw_safe = soft_clamp(kappa_raw, -10.0, 4.0)
        kappa = F.softplus(kappa_raw_safe + np.log(self.args.kappa_init)) + 1e-06
        return (mu, kappa)

    def simple_vmf_sample(self, mu, kappa):
        """Soft normalization can also be used."""
        noise = torch.randn_like(mu)
        noise = F.normalize(noise, p=2, dim=-1)
        kappa_values = kappa.squeeze(-1)
        kappa_safe = torch.clamp(kappa_values, max=20.0)
        weight = torch.sigmoid(kappa_safe * 0.5)
        sample = weight.unsqueeze(-1) * mu + (1 - weight.unsqueeze(-1)) * noise
        return sample * self.radius

    def compute_vmf_entropy_approximate(self, kappa):
        """ H ≈ -κ"""
        kappa_values = kappa.squeeze(-1)
        vmf_entropy_approx = -kappa_values
        return vmf_entropy_approx

    def compute_kappa_regularization(self, kappa):
        kappa_values = kappa.squeeze(-1)
        lower_penalty = F.relu(self.args.kappa_min - kappa_values) ** 2
        upper_penalty = F.relu(kappa_values - self.args.kappa_max) ** 2
        mean_kappa = kappa_values.mean()
        kappa_variance = kappa_values.var()
        exploration_protection = F.relu(mean_kappa - 3.0) ** 2 + F.relu(0.1 - kappa_variance) ** 2
        kappa_safe = torch.clamp(kappa_values, min=0.1)
        kappa_entropy_reg = -torch.log(kappa_safe + 1e-08).mean()
        reg_loss = (lower_penalty + upper_penalty).mean() + 0.3 * exploration_protection + 0.05 * kappa_entropy_reg
        return reg_loss

    def get_action(self, x, return_vmf_entropy=False):
        mu, kappa = self.forward(x)
        d_hat = self.simple_vmf_sample(mu, kappa)
        action = d_hat * self.action_scale + self.action_bias
        vmf_entropy = self.compute_vmf_entropy_approximate(kappa)
        mean = mu * self.action_scale + self.action_bias
        if return_vmf_entropy:
            return (action, vmf_entropy, mean, kappa)
        else:
            return (action, vmf_entropy, mean)


class GACReferenceAgent:
    """Torch implementation of the pinned learner; driver clocks stay explicit."""

    def __init__(self, seed, observation_dim, action_low, action_high, config=None, device='cpu'):
        self.config = config or GACReferenceConfig()
        self.config.validate()
        self.device = torch.device(device)
        low, high = np.asarray(action_low, dtype='float32'), np.asarray(action_high, dtype='float32')
        if (low.ndim != 1 or high.shape != low.shape or not len(low) or observation_dim < 1
                or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low)):
            raise ValueError('Invalid vector observation/action spaces.')
        self.observation_dim, self.action_dim = observation_dim, len(low)
        self.action_low, self.action_high = low.copy(), high.copy()
        self.cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        env = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(observation_dim,)),
                              single_action_space=SimpleNamespace(shape=low.shape, low=low, high=high))
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.manual_seed(seed)
            self.actor = GACActor(env, self.config).to(self.device)
            self.qf1 = SoftQNetwork(env).to(self.device)
            self.qf2 = SoftQNetwork(env).to(self.device)
            self.qf1_target = SoftQNetwork(env).to(self.device)
            self.qf2_target = SoftQNetwork(env).to(self.device)
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
            self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.config.q_lr)
            self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.config.policy_lr)
            self.rng_cpu = torch.get_rng_state().clone()
            self.rng_cuda = torch.cuda.get_rng_state(self.cuda_devices[0]).clone() if self.cuda_devices else None
        self.critic_updates = self.actor_updates = self.target_updates = 0
        self.last_global_step = None

    @contextmanager
    def _rng(self, seed=None):
        with torch.random.fork_rng(devices=self.cuda_devices):
            if seed is None:
                torch.set_rng_state(self.rng_cpu)
                if self.cuda_devices: torch.cuda.set_rng_state(self.rng_cuda, self.cuda_devices[0])
            else:
                torch.manual_seed(seed)
            yield
            if seed is None:
                self.rng_cpu = torch.get_rng_state().clone()
                if self.cuda_devices: self.rng_cuda = torch.cuda.get_rng_state(self.cuda_devices[0]).clone()

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def sample_actions(self, observations, seed=None):
        values = self._tensor(observations)
        single = values.ndim == 1
        if single: values = values[None]
        with torch.no_grad(), self._rng(seed):
            actions, _, _ = self.actor.get_action(values)
        actions = actions.cpu().numpy()
        return actions[0] if single else actions

    def update(self, batch, global_step):
        if (type(global_step) is not int or global_step <= self.config.learning_starts
                or self.last_global_step is not None and global_step <= self.last_global_step):
            raise ValueError('Use increasing zero-based loop steps strictly after learning_starts.')
        required = ('observations','actions','next_observations','rewards','dones')
        data = SimpleNamespace(**{key:self._tensor(batch[key]) for key in required})
        size = data.observations.shape[0]
        if (size < 2 or data.observations.shape != (size,self.observation_dim)
                or data.next_observations.shape != data.observations.shape or data.actions.shape != (size,self.action_dim)
                or data.rewards.numel()!=size or data.dones.numel()!=size
                or not all(torch.isfinite(getattr(data,key)).all() for key in required)):
            raise ValueError('Invalid finite GAC transition batch; singleton variance is unsupported.')
        if not torch.all((data.dones==0)|(data.dones==1)):
            raise ValueError('Expected binary terminal masks.')
        args = self.config
        with self._rng():
            with torch.no_grad():
                next_state_actions, next_state_vmf_entropy, _ = self.actor.get_action(data.next_observations)
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) + args.alpha * next_state_vmf_entropy.unsqueeze(-1)
                next_q_value = data.rewards.flatten() + (1-data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)
            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()
            self.critic_updates += 1
            info = dict(qf_loss=float(qf_loss.detach()), qf1_loss=float(qf1_loss.detach()), qf2_loss=float(qf2_loss.detach()))
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, vmf_entropy, _, kappa = self.actor.get_action(data.observations, return_vmf_entropy=True)
                    qf1_pi, qf2_pi = self.qf1(data.observations, pi), self.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    sac_actor_loss = ((-args.alpha*vmf_entropy.unsqueeze(-1))-min_qf_pi).mean()
                    kappa_reg_loss = self.actor.compute_kappa_regularization(kappa)
                    actor_loss = sac_actor_loss + args.kappa_reg_weight*kappa_reg_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
                    self.actor_optimizer.step()
                    self.actor_updates += 1
                info.update(actor_loss=float(actor_loss.detach()),sac_actor_loss=float(sac_actor_loss.detach()),kappa_reg_loss=float(kappa_reg_loss.detach()))
            if global_step % args.target_network_frequency == 0:
                for param,target in zip(self.qf1.parameters(),self.qf1_target.parameters()):
                    target.data.copy_(args.tau*param.data+(1-args.tau)*target.data)
                for param,target in zip(self.qf2.parameters(),self.qf2_target.parameters()):
                    target.data.copy_(args.tau*param.data+(1-args.tau)*target.data)
                self.target_updates += 1
        self.last_global_step = global_step
        return info

    def state_dict(self):
        return copy.deepcopy(dict(config=asdict(self.config),observation_dim=self.observation_dim,
            action_low=self.action_low.tolist(),action_high=self.action_high.tolist(),
            models={name:getattr(self,name).state_dict() for name in ('actor','qf1','qf2','qf1_target','qf2_target')},
            q_optimizer=self.q_optimizer.state_dict(),actor_optimizer=self.actor_optimizer.state_dict(),
            rng_cpu=self.rng_cpu,rng_cuda=self.rng_cuda,critic_updates=self.critic_updates,
            actor_updates=self.actor_updates,target_updates=self.target_updates,last_global_step=self.last_global_step))

    def load_state_dict(self, state):
        if (state['config'] != asdict(self.config) or state['observation_dim'] != self.observation_dim
                or state['action_low'] != self.action_low.tolist() or state['action_high'] != self.action_high.tolist()):
            raise ValueError('Checkpoint architecture/configuration differs.')
        expected_models = {'actor','qf1','qf2','qf1_target','qf2_target'}
        if set(state['models']) != expected_models:
            raise ValueError('Checkpoint model set differs.')
        if bool(self.cuda_devices)!=(state['rng_cuda'] is not None):
            raise ValueError('Cross-device RNG continuation is unsupported; use explicit-seed policy probes instead.')
        for name,value in state['models'].items():
            template=getattr(self,name).state_dict()
            if set(template)!=set(value) or any(template[key].shape!=value[key].shape or template[key].dtype!=value[key].dtype for key in template):
                raise ValueError('Checkpoint model tensor structure differs.')
            getattr(self,name).load_state_dict(value,strict=True)
        self.q_optimizer.load_state_dict(state['q_optimizer']);self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.rng_cpu=state['rng_cpu'].cpu().clone()
        self.rng_cuda=state['rng_cuda'].cpu().clone() if state['rng_cuda'] is not None else None
        for name in ('critic_updates','actor_updates','target_updates','last_global_step'):setattr(self,name,state[name])
