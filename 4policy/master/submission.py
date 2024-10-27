
from __future__ import annotations
from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from EnterpriseMAE_CC4 import EnterpriseMAE

# Import your custom agents here.
from Ray_BlueAgent import HMARLBlueAgent

class Submission:

    # Submission name
    NAME: str = "cyborg"

    # Name of your team
    TEAM: str = "Blue"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "H-MARL"

    # Use this function to define your agents.
    # AGENTS: dict[str, BaseAgent] = {
    #     f"blue_agent_{agent}": Ray_BlueAgent(name=f"Agent{agent}") for agent in range(5)
    # }
    AGENTS = {}
    for a in range(5):
        AGENTS[f"blue_agent_{a}_master"]  = HMARLBlueAgent(name=f"Agent{a}_master")

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)

