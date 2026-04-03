"""Abstract base class for reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for all reward functions.

    Subclasses must implement ``compute``, ``latex``, and ``terms``.
    ``__call__`` delegates to ``compute`` for backward compatibility with
    code that invokes ``reward_fn(obs, action, next_obs, info)``.
    """

    @abstractmethod
    def compute(self, obs, action, next_obs, info) -> tuple[float, dict[str, float]]:
        """Return (total_reward, {term_name: value})."""

    @property
    @abstractmethod
    def latex(self) -> str:
        """LaTeX formula describing the reward."""

    @property
    @abstractmethod
    def terms(self) -> dict[str, str]:
        """Mapping of term name → human description."""

    @property
    def description(self) -> str:
        """Human-readable description. Defaults to class docstring."""
        return type(self).__doc__ or ""

    def __call__(self, *args):
        return self.compute(*args[:4])
