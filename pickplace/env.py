import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from pymunk.vec2d import Vec2d
import pybullet as p
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.utils import DrawOptions


DEFAULT_VARIATIONS = (
    'agent.start_position',
    'block.start_position',
    'block.angle',
)

class PickPlace(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'render_fps': 10,
        }

    reward_range = (0.0, 1.0)

    def __init__(self, gui=False):

        self._seed = None

        physicsClient = p.connect(p.GUI )
        


