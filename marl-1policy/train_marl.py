import inspect
import time
import os

from statistics import mean, stdev
from typing import Any
from rich import print
import numpy as np

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from EnterpriseMAE_CC4 import EnterpriseMAE

from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPO, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import register_env

from action_mask_model_CC4 import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

from typing import Dict, Tuple

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

ModelCatalog.register_custom_model(
    "my_model", TorchActionMaskModel
)


def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    cyborg = EnterpriseMAE(cyborg)
    return cyborg


NUM_AGENTS = 5

# mapping to the policy directory name
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]


# register_env(name="CC4", env_creator=lambda config: ParallelPettingZooEnv(env_creator_CC4(config)))
register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
env = env_creator_CC4({})


# Note:     will allow different action space sizes but not different observation space sizes in one property
#           current implementation may cause issues - seems to want all same size???
algo_config = (
    PPOConfig()
    .framework("torch")
    .debugging(logger_config={"logdir":"logs/train_marl", "type":"ray.tune.logger.TBXLogger"})
    .environment(env="CC4")
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #.resources(num_gpus=1)  # export CUDA_VISIBLE_DEVICES=0,1
    .experimental(
        _disable_preprocessor_api=True,
    )
    .rollouts(
        batch_mode="complete_episodes",
        num_rollout_workers=30, # for debugging, set this to 0 to run in the main thread
    )
    .training(
        model={"custom_model": "my_model"},
        sgd_minibatch_size=32768, # default 128
        train_batch_size=1000000, # default 4000
        )
    .multi_agent(
        policies={
            ray_agent: PolicySpec(
                policy_class=PPOTorchPolicy,
                observation_space=env.observation_space(cyborg_agent),
                action_space=env.action_space(cyborg_agent),
                config={"entropy_coeff": 0.001},
            )
            for cyborg_agent, ray_agent in POLICY_MAP.items()
        },
        policy_mapping_fn=policy_mapper,
    )
)

model_dir = "models/train_marl"

check_env(env)
algo = algo_config.build()

# if need to restore, hopefully not
# checkpoint_path = "models/train_CC4/iter_155"
# algo.restore(checkpoint_path)

for i in range(200):
    iteration = i # for restore, adjust iter, overwise you will  overwrite old models, e.g.  i + 156
    train_info = algo.train()
    print("\nIteration:", i, train_info)
    model_dir_crt = os.path.join(model_dir, "iter_"+str(iteration))
    print("\nSaving model in:", model_dir_crt)
    algo.save(model_dir_crt)

algo.save(model_dir_crt)


