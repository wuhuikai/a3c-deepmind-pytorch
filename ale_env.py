import sys
import time

import collections

import cv2
import numpy as np
from ale_python_interface import ALEInterface

from rl import Environment

class ALE(Environment):
    """Arcade Learning Environment.
    """
    def __init__(self, rom_path, n_last_screens=4, frame_skip=4, treat_life_lost_as_terminal=True,
                 crop_or_scale='scale', max_start_nullops=30, record_screen_dir=None, render=False, max_episode_length=None, max_time=None):
        self.frame_skip = frame_skip
        self.n_last_screens = n_last_screens
        self.treat_life_lost_as_terminal = treat_life_lost_as_terminal
        self.crop_or_scale = crop_or_scale
        self.max_start_nullops = max_start_nullops
        self.max_episode_length = max_episode_length
        self.max_time = max_time

        ale = ALEInterface()
        # Use numpy's random state
        seed = np.random.randint(0, 2 ** 16)
        ale.setInt(b'random_seed', seed)
        ale.setFloat(b'repeat_action_probability', 0.0)
        ale.setBool(b'color_averaging', False)
        
        if record_screen_dir is not None:
            ale.setString(b'record_screen_dir', str.encode(record_screen_dir))
        
        if render:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                ale.setBool(b'sound', False)  # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                ale.setBool(b'sound', True)
            ale.setBool(b'display_screen', True)
        
        ale.loadROM(str.encode(rom_path))

        self.ale = ale
        self.__exceed_max = False
        self.legal_actions = ale.getMinimalActionSet()
        self.reset()

    def current_screen(self):
        # Max of two consecutive frames
        rgb_img = np.maximum(self.ale.getScreenRGB(), self.last_raw_screen)
        # Make sure the last raw screen is used only once
        self.last_raw_screen = None

        # RGB -> Luminance
        img = rgb_img[:, :, 0] * 0.2126 + rgb_img[:, :, 1] * \
            0.0722 + rgb_img[:, :, 2] * 0.7152
        img = img.astype(np.uint8)
        if img.shape == (250, 160):
            raise RuntimeError("This ROM is for PAL. Please use ROMs for NTSC")
        if self.crop_or_scale == 'crop':
            # Shrink (210, 160) -> (110, 84)
            img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_LINEAR)
            # Crop (110, 84) -> (84, 84)
            bottom_crop = 8
            img = img[110 - 84 - bottom_crop: 110 - bottom_crop, :]
        elif self.crop_or_scale == 'scale':
            img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        else:
            raise RuntimeError('Unknow crop_or_scale: [{}], must be one of [crop|scale]'.format(self.crop_or_scale))
        return img

    @property
    def current_raw_screen(self):
        return self.ale.getScreenRGB()

    @property
    def state(self):
        return list(self.last_screens)

    @property
    def is_terminal(self):
        if self.max_time and time.time() - self.time > self.max_time:
            self.__exceed_max = True
            return True
        if self.max_episode_length and self.step > self.max_episode_length:
            self.__exceed_max = True
            return True
        if self.treat_life_lost_as_terminal:
            return self.lives_lost or self.ale.game_over()
        else:
            return self.ale.game_over()

    @property
    def exceed_max(self):
        return self.__exceed_max

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    def receive_action(self, action):
        self.step += 1

        rewards = []
        for i in range(self.frame_skip):
            # Last screeen must be stored before executing the 4th action
            if i == self.frame_skip-1:
                self.last_raw_screen = self.ale.getScreenRGB()

            rewards.append(self.ale.act(self.legal_actions[action]))

            # Check if lives are lost
            if self.lives > self.ale.lives():
                self.lives_lost = True
            else:
                self.lives_lost = False
            self.lives = self.ale.lives()

            if self.is_terminal:
                break

        # We must have last screen here unless it's terminal
        if not self.is_terminal:
            self.last_screens.append(self.current_screen())

        self._reward = sum(rewards)

        return self._reward

    def reset(self):
        if self.ale.game_over() or self.__exceed_max:
            self.ale.reset_game()

        if self.max_start_nullops > 0:
            n_nullops = np.random.randint(0, self.max_start_nullops + 1)
            for _ in range(n_nullops):
                self.ale.act(0)

        self._reward = 0

        self.last_raw_screen = self.ale.getScreenRGB()

        self.last_screens = collections.deque(
            [np.zeros((84, 84), dtype=np.uint8)] * 3 +
            [self.current_screen()],
            maxlen=self.n_last_screens)

        self.lives_lost = False
        self.lives = self.ale.lives()

        self.step = 0
        self.time = time.time()
        self.__exceed_max = False