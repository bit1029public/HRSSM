import gym
import numpy as np
from envs.exceptions import UnknownTaskError
import mani_skill2.envs

MANISKILL_TASKS = {
    'lift-cube': dict(
        env='LiftCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'pick-cube': dict(
        env='PickCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'stack-cube': dict(
        env='StackCube-v0',
        control_mode='pd_ee_delta_pos',
    ),
    'pick-ycb': dict(
        env='PickSingleYCB-v0',
        control_mode='pd_ee_delta_pose',
    ),
    'turn-faucet': dict(
        env='TurnFaucet-v0',
        control_mode='pd_ee_delta_pose',
    ),
    'peg-insertion-side': dict(
        env='PandaAvoidObstacles-v0',
        control_mode='pd_ee_delta_pose',
    )
}


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )
        self._action_repeat = config.action_repeat
        self._full_log = not config.simple_log

    @property
    def observation_space(self):
        if self._full_log:
            spaces = {
                "obs": self.env.observation_space,
                "image": gym.spaces.Box(
                    0, 255, self.render().shape, dtype=np.uint8 
                ),
                "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                # "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            }
        else:
            spaces = {
                "obs": self.env.observation_space,
                "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                # "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            }
        return gym.spaces.Dict(spaces)
    
    def reset(self):
        if self._full_log:
            obs = {
                "obs": self.env.reset(),
                "image": self.render(),
                "is_first": True,
                "is_terminal": False
            }
        else:
            obs = {
                "obs": self.env.reset(),
                "is_first": True,
                "is_terminal": False
            }
        return obs
    
    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        if self._full_log:
            obs = {
                "obs": obs,
                "image": self.render(),
                "is_first": False,
                "is_terminal": False
            }
        else:
            obs = {
                "obs": obs,
                "is_first": False,
                "is_terminal": False
            }
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self):
        return self.env.render(mode='cameras')


def make_env(task, config):
    """
    Make ManiSkill2 environment.
    """
    task = "-".join(task.split("_"))
    if task not in MANISKILL_TASKS:
        raise UnknownTaskError(task)
    task_cfg = MANISKILL_TASKS[task]
    env = gym.make(
        task_cfg['env'],
        obs_mode='state',
        control_mode=task_cfg['control_mode'],
        render_camera_cfgs=dict(width=64, height=64),
    )
    env = ManiSkillWrapper(env, config=config)
    return env
