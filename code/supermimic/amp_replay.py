"""AMP replay with clean policy actions and historical velocity commands."""

from __future__ import annotations

import torch

from .replay import ReplayBuffer
from .task import flat_to_state


class AMPReplay(ReplayBuffer):
    """Keep physical executed actions separate from the actor observation history.

    The inherited a field is the executed (possibly noisy) action for world-model
    fitting. AMP actor histories instead use the explicitly logged clean previous
    action, paired with the command and context of each historical state.
    """

    def __init__(self, num_envs, capacity, xi_dim, device):
        super().__init__(num_envs, capacity, xi_dim, device)
        self.command = torch.zeros(capacity, num_envs, 3, device=device)
        self.previous_clean = torch.zeros(capacity, num_envs, 29, device=device)
        self.clean_action = torch.zeros(capacity, num_envs, 29, device=device)

    @torch.no_grad()
    def add_amp(self, command, previous_clean, clean_action):
        """Call immediately after add(), with fields of that same transition."""
        if self.size == 0:
            raise RuntimeError("add_amp() must follow ReplayBuffer.add()")
        row = (self.ptr - 1) % self.L
        self.command[row] = command
        self.previous_clean[row] = previous_clean
        self.clean_action[row] = clean_action

    @torch.no_grad()
    def sample_starts_amp(self, batch, history, amp_task):
        """Return s0, xi, current command, clean observation history, prior action.

        Histories are oldest first and never read across an episode or past the
        oldest buffered state. Missing history repeats the earliest available
        state of that episode, including its command, context and clean action.
        """
        if self.size < 1:
            raise RuntimeError("Cannot sample AMP starts from an empty replay buffer")
        if history < 1:
            raise ValueError("history must be at least one")
        env = torch.randint(0, self.n, (batch,), device=self.device)
        age = torch.randint(0, self.size, (batch,), device=self.device)
        row = self._age_index(age)
        episode = self.ep[row, env]
        back = torch.arange(history - 1, -1, -1, device=self.device)
        ages = age[:, None] + back
        rows = self._age_index(ages.clamp(max=self.size - 1))
        envs = env[:, None].expand(-1, history)
        same = (self.ep[rows, envs] == episode[:, None]) & (ages < self.size)
        # The current row is always in the selected episode, so at least one
        # entry is available even immediately after a reset.
        first = same.long().argmax(-1)
        padding_row = rows.gather(1, first[:, None]).expand_as(rows)
        rows = torch.where(same, rows, padding_row)
        history_state = flat_to_state(self.s[rows, envs])
        obs = amp_task.observation(
            history_state,
            self.previous_clean[rows, envs],
            self.command[rows, envs],
            self.xi[rows, envs],
        )
        return (
            self.s[row, env],
            self.xi[row, env],
            self.command[row, env],
            obs,
            self.previous_clean[row, env],
        )

    @torch.no_grad()
    def sample_transitions(self, batch):
        """Real physical (s,s1) pairs, including terminal pre-reset successors."""
        if self.size < 1:
            raise RuntimeError("Cannot sample transitions from an empty replay buffer")
        env = torch.randint(0, self.n, (batch,), device=self.device)
        age = torch.randint(0, self.size, (batch,), device=self.device)
        row = self._age_index(age)
        return self.s[row, env], self.s1[row, env]
