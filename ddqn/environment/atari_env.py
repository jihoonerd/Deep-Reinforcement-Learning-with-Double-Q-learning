import gym
import numpy as np

from ddqn.utils.atari_wrappers import make_atari, wrap_deepmind


class Environment:

    def __init__(self, env_id, train=True):
        clip_rewards = True if train else False
        self.env = wrap_deepmind(make_atari(env_id), clip_rewards=clip_rewards, frame_stack=True)
    
    def reset(self):
        reset_state = self.env.reset()
        return np.array(reset_state)
    
    def render(self):
        return self.env.render(mode='rgb_array')

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return np.array(next_state), reward, done, info

    def get_action_space_size(self):
        return self.env.action_space.n
