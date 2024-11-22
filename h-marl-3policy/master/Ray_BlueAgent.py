import os
from CybORG.Agents import BaseAgent
from gym import Space
import subprocess
import numpy as np
import torch

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.models import ModelCatalog
from action_mask_model_CC4 import TorchActionMaskModel

ModelCatalog.register_custom_model(
    "my_model", TorchActionMaskModel
)

from serializable_policy import SerializablePolicy

class HMARLBlueAgent(BaseAgent):
    def __init__(self, name: str = None):
        super().__init__(name)
       
        
        # each blue agent has its own policy
        if "master" in self.name:
            pkl_cp_dir = os.path.join("saved_policies/master/iter_49/policies/", self.name) # for evaluation
            # return # to train, just return
        else:
            pkl_cp_dir = os.path.join("saved_policies/sub/iter_49/policies/", self.name)
    
        print("\nLoading Serializable blue agent model from ", pkl_cp_dir)
        self.policy = SerializablePolicy.from_checkpoint(pkl_cp_dir)


    def get_action(self, observation: dict, action_space: Space):

        # Use the restored policy for serving actions.
        action, state_out, extra = self.policy.compute_single_action(obs=observation)
    
        return action

