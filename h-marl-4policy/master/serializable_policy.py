from ray.rllib.policy import Policy

# Quick wrapper to support Policy object serialization w/ Ray
# TODO: Need to understand why they aren't supported natively
class SerializablePolicy:
    def __init__(self, policy):
        self.policy = policy
    
    def compute_single_action(self, obs):
        return self.policy.compute_single_action(obs)
    
    @staticmethod
    def from_checkpoint(checkpoint_dir):
        p = Policy.from_checkpoint(checkpoint_dir)
        return SerializablePolicy(p)
    
    @staticmethod
    def from_state(state):
        p = Policy.from_state(state)
        return SerializablePolicy(p)
    
    def get_state(self):
        return self.policy.get_state()
    
    def __reduce__(self):
        return (SerializablePolicy.from_state, (self.get_state(),))