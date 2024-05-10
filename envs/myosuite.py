import numpy as np
import gym
from envs.exceptions import UnknownTaskError


MYOSUITE_TASKS = {
    'myo-finger-reach': 'myoFingerReachFixed-v0',
    'myo-finger-reach-hard': 'myoFingerReachRandom-v0',
    'myo-finger-pose': 'myoFingerPoseFixed-v0',
    'myo-finger-pose-hard': 'myoFingerPoseRandom-v0',
    'myo-hand-reach': 'myoHandReachFixed-v0',
    'myo-hand-reach-hard': 'myoHandReachRandom-v0',
    'myo-hand-pose': 'myoHandPoseFixed-v0',
    'myo-hand-pose-hard': 'myoHandPoseRandom-v0',
    'myo-hand-obj-hold': 'myoHandObjHoldFixed-v0',
    'myo-hand-obj-hold-hard': 'myoHandObjHoldRandom-v0',
    'myo-hand-key-turn': 'myoHandKeyTurnFixed-v0',
    'myo-hand-key-turn-hard': 'myoHandKeyTurnRandom-v0',
    'myo-hand-pen-twirl': 'myoHandPenTwirlFixed-v0',
    'myo-hand-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}


class MyoSuiteWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat):
        self.env = env # TODO super().__init__(env)
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = getattr(self.env, 'spec', None)
        self.env = env
        self.camera_id = 'hand_side_inter'
        self._action_repeat = action_repeat
  
    @property
    def observation_space(self):
        spaces = {
            "obs": gym.spaces.Box(low=self.env.observation_space.low.astype(np.float32), high=self.env.observation_space.high.astype(np.float32), dtype=np.float32),
            # "image": gym.spaces.Box(
            #     0, 255, self.render().shape, dtype=np.uint8
            # ),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            # "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
        }
        return gym.spaces.Dict(spaces)
    
    def reset(self, **kwargs):
        obs = {
            "obs": super().reset(**kwargs).astype(np.float32),
            # "image": self.render(),
            "is_first": True,
            "is_terminal": False
        }
        return obs
    
    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        obs = {
            "obs": obs,
            # "image": self.render(),
            "is_first": False,
            "is_terminal": False
        }
        info['success'] = info['solved']
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.sim.renderer.render_offscreen(
            width=384, height=384, camera_id=self.camera_id
        ).copy()
        # return self.env.render(
        #     offscreen=True, resolution=(384, 384), camera_id=self.camera_id
        # ).copy()


def make_env(task, config):
    """
    Make Myosuite environment.
    """
    task = "-".join(task.split("_"))
    if not task in MYOSUITE_TASKS:
        raise UnknownTaskError(task)
    import myosuite
    env = gym.make(MYOSUITE_TASKS[task])
    env = MyoSuiteWrapper(env, action_repeat=config.action_repeat)
    return env
