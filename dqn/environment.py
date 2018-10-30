import random

from ShipEnv.shipenv import ShipEnv

try:
    import gym
except ModuleNotFoundError as e:
    print("[*] Warning: No module named 'gym'")

from .utils import rgb2gray, resize


class Environment(object):
    def __init__(self, config):
        if config.env_name.startswith("Ship"):
            self.env = ShipEnv(screen_size=(300, 300), fps=6000)

        else:
            self.env = gym.make(config.env_name)

        screen_width, screen_height = config.screen_width, config.screen_height
        self.action_repeat, self.random_start = config.action_repeat, config.random_start
        self.dims = (screen_width, screen_height)

        self.display = config.display
        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        # if self.lives == 0:
        #     self._screen = self.env.reset()
        self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, self.info = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        # Pre-processing: Channel gray scale & Downsampling
        return resize(rgb2gray(self._screen) / 255., self.dims)
        # return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        try:
            return self.env.env.ale.lives()
        # For Ship environment
        except Exception as e:
            return 1

    @property
    def state(self):
        return self.screen, self.reward, self.terminal, self.info

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()


class GymEnvironment(Environment):
    def __init__(self, config):
        super(GymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        cumulated_reward = 0
        # start_lives = self.lives
        start_lives = 1

        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulated_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Environment):
    def __init__(self, config):
        super(SimpleGymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        self._step(action)

        self.after_act(action)
        return self.state
