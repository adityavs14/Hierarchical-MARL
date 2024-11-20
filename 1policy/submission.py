
from __future__ import annotations
from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from EnterpriseMAE_CC4 import EnterpriseMAE

# Import your custom agents here.
from Ray_BlueAgent import MARLAgent

class Submission:

    # Submission name
    NAME: str = "Blue MARL"

    # Name of your team
    TEAM: str = "Blue"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "PPO MARL independent learners with masking and NoOp extra ticks"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": MARLAgent(name=f"Agent{agent}") for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)

