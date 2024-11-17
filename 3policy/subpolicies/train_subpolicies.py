import inspect
import time
import os

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

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from action_mask_model_CC4 import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
import gymnasium

ModelCatalog.register_custom_model(
    "my_model", TorchActionMaskModel
)


class CCPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        self.config = config
        # just in case we are interested, the policy if is in self.config["__policy_id"]


    def handle_extra_ticks(self, postprocessed_batch):
        rewards = None

        # shifting master rewards by -1
        if "id" not in postprocessed_batch["obs"]:
            return postprocessed_batch

        if "rewards" not in postprocessed_batch:
            return postprocessed_batch

        if len(postprocessed_batch["rewards"]) <= 1:
            return postprocessed_batch

        rewards = postprocessed_batch["rewards"][1:]
        rewards = np.concatenate((rewards,[0]))
        postprocessed_batch["rewards"] = rewards

        return postprocessed_batch


    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # handle extra ticks first, update rewards
        sample_batch = self.handle_extra_ticks(sample_batch)

        # continue with the default postprocessing (i.e., computing advantages)
        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
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
# POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}
POLICY_MAP = {}
for i in range(NUM_AGENTS):
    POLICY_MAP[f"blue_agent_{i}_master"]  = f"Agent{i}_master"
    POLICY_MAP[f"blue_agent_{i}_investigate"]  = f"Agent{i}_investigate"
    POLICY_MAP[f"blue_agent_{i}_recover"]  = f"Agent{i}_recover"

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

    OBSERVATION_SPACE[f"Agent{i}_investigate"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_investigate']})
    ACTION_SPACE[f"Agent{i}_investigate"] = env.action_space(f"blue_agent_{i}")

    OBSERVATION_SPACE[f"Agent{i}_recover"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_recover']})
    ACTION_SPACE[f"Agent{i}_recover"] = env.action_space(f"blue_agent_{i}")


# Note:     will allow different action space sizes but not different observation space sizes in one property
#           current implementation may cause issues - seems to want all same size???
algo_config = (
    PPOConfig()
    .framework("torch")
    .debugging(logger_config={"logdir":"logs/train_subpolicies", "type":"ray.tune.logger.TBXLogger"})
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
        sgd_minibatch_size = 32768, # default 128
        train_batch_size = 1000000, # default 4000
        )
    .multi_agent(
        policies={
            ray_agent: PolicySpec(
                policy_class = CCPPOTorchPolicy,
                observation_space = OBSERVATION_SPACE[ray_agent],
                action_space = ACTION_SPACE[ray_agent],
                config = {"entropy_coeff": 0.001},
            )
            for ray_agent in OBSERVATION_SPACE
        },
        policy_mapping_fn=policy_mapper,
    )
)

model_dir = "saved_policies/sub_v1"

check_env(env)
algo = algo_config.build()

# if need to restore
# checkpoint_path = "models/train_CC4/iter_10"
# algo.restore(checkpoint_path)

for i in range(50):
    iteration = i # for restore, adjust iter, overwise you will  overwrite old models, e.g.  i + 156
    train_info = algo.train()
    print("\nIteration:", i, train_info)
    model_dir_crt = os.path.join(model_dir, "iter_"+str(iteration))
    print("\nSaving model in:", model_dir_crt)
    algo.save(model_dir_crt)

algo.save(model_dir_crt)


