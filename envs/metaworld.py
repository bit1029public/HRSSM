import numpy as np
import gym
from envs.exceptions import UnknownTaskError

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self._camera = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False
        self._action_repeat = config.action_repeat
        self._full_log = not config.simple_log
        self._render_size = config.size

    @property
    def observation_space(self):
        spaces = {
            "obs": gym.spaces.Box(low=self.env.observation_space.low.astype(np.float32), high=self.env.observation_space.high.astype(np.float32), dtype=np.float32),
            "image": gym.spaces.Box(
                0, 255, self.render().shape, dtype=np.uint8
            ),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return gym.spaces.Dict(spaces)
    
    # @property 
    # def action_space(self):
    #     spec = self.action_spec()
    #     return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs = {
            "obs": super().reset(**kwargs).astype(np.float32),
            "image": self.render(),
            "is_first": True,
            "is_last": False,
            "is_terminal": False
        }
        # self.env.step(np.zeros(self.env.action_space.shape))
        return obs

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        obs = {
            "obs": obs,
            "image": self.render(),
            "is_first": False,
            "is_last": False,
            "is_terminal": False
        }
        
        # obs = obs.astype(np.float32)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        # return self.env.render(
        #     offscreen=True, resolution=self._render_size, camera_name=self._camera
        # ).copy()

        return self.env.sim.render(
                *self._render_size, mode="offscreen", camera_name=self._camera
            )


def make_env(task, config):
    """
    Make Meta-World environment.
    """
    task = "-".join(task.split("_"))
    env_id = task + "-v2-goal-observable"
    if env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise UnknownTaskError(task)
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=config.seed)
    env = MetaWorldWrapper(env, config=config)
    return env
