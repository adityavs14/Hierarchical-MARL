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
from ray.rllib.algorithms.algorithm import Algorithm

from action_mask_model_CC4 import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from typing import Dict, Tuple
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import gymnasium
from ray.rllib.utils.annotations import override
# from Ray_BlueAgent import Ray_BlueAgent


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


def algo_builder(num_workers = 0):


    NUM_AGENTS = 5

    # mapping to the policy directory name
    POLICY_MAP = {}
    for i in range(NUM_AGENTS):
        POLICY_MAP[f"blue_agent_{i}_master"]  = f"Agent{i}_master"

    def policy_mapper(agent_id, episode, worker, **kwargs):
        return POLICY_MAP[agent_id]


    # register_env(name="CC4", env_creator=lambda config: ParallelPettingZooEnv(env_creator_CC4(config)))
    register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))
    env = env_creator_CC4({})


    OBSERVATION_SPACE = {}
    ACTION_SPACE = {}
    for i in range(NUM_AGENTS):
        OBSERVATION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.Dict({'action_mask': gymnasium.spaces.multi_discrete.MultiDiscrete([2,2]),'observations':env.observation_space(f'blue_agent_{i}')['observations'], 'id':gymnasium.spaces.discrete.Discrete(1)})
        ACTION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.discrete.Discrete(2)
        

    # Note:     will allow different action space sizes but not different observation space sizes in one property
    #           current implementation may cause issues - seems to want all same size???

    algo_config = (
        PPOConfig()
        .framework("torch")
        .debugging(logger_config={"logdir":"logs/train_master", "type":"ray.tune.logger.TBXLogger"})
        .environment(env="CC4")
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #.resources(num_gpus=1)  # export CUDA_VISIBLE_DEVICES=0,1
        .experimental(
            _disable_preprocessor_api=True,
        )
        .rollouts(
            batch_mode="complete_episodes",
            num_rollout_workers=num_workers, # for debugging, set this to 0 to run in the main thread
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
                    observation_space=OBSERVATION_SPACE[ray_agent],
                    action_space=ACTION_SPACE[ray_agent],
                    config={"entropy_coeff": 0.001},
                )
                for ray_agent in OBSERVATION_SPACE
            },
            policy_mapping_fn=policy_mapper,
        )
    )


    check_env(env)
    algo = algo_config.build()
    return algo

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("CybORG Master Training Script")
    parser.add_argument("num_workers", type=int, default=0, help="Set the number of rollout workers for training")
    args = parser.parse_args()
    
    algo = algo_builder(num_workers=args.num_workers)

    # to restore
    # checkpoint_path = "models/train_CC4/iter_50"
    # algo.restore(checkpoint_path)
    model_dir = "saved_policies/master"

    for i in range(50):
        iteration = i # for restore, adjust iter, overwise you will  overwrite old models, e.g.  i + 156
        train_info = algo.train()
        print("\nIteration:", i, train_info)
        model_dir_crt = os.path.join(model_dir, "iter_"+str(iteration))
        print("\nSaving model in:", model_dir_crt)
        algo.save(model_dir_crt)

    algo.save(model_dir_crt)


