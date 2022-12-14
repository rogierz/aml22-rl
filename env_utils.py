import gymnasium as gym
import numpy as np

class CustomHopperWrapper(gym.Wrapper):
    """Custom Wrapper for the Hopper-v4 gym environment

    - Adds domain randomization capabilities
    - Handles source and target domains

    See https://gymnasium.farama.org/api/wrappers/
    """

    def __init__(self, env, domain):
        super().__init__(env)
        assert self.env.unwrapped.spec.id == 'Hopper-v4'

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.env.model.body_mass[1] -= 1.0

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        raise NotImplementedError()

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.env.model.body_mass[1:] )
        return masses

    def set_parameters(self, masses):
        """Set each hopper link's mass to a new value"""
        self.env.model.body_mass[1:] = masses


def make_env(domain, render_mode=None):
    """Returns the wrapped Hopper-v4 environment"""
    assert domain in ['source', 'target']

    env = gym.make('Hopper-v4', render_mode=render_mode)
    env = gym.wrappers.StepAPICompatibility(env, output_truncation_bool=False)  # Retro-compatibility for stable-baselines3
    env = CustomHopperWrapper(env, domain=domain)  # Use custom implementation for source/target variants
    
    return env