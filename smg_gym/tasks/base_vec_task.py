# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from typing import Dict, Any, Tuple, Union
from types import SimpleNamespace
import abc
from abc import ABC

import time
import gym
from gym import spaces

import numpy as np
import torch

from isaacgym import gymapi


class Env(ABC):
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int,  headless: bool):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        # Add an instance variable to track the elapsed time
        self.x=0
        self.elapsed_time = 0.0
        self.gravity_change_interval = 1.0  # Change gravity every 1 second 
        self.gravity = config['sim']['gravity']
        self.config= config 

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", "cuda:0")

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self._num_environments = config["env"]["num_envs"]
        self._num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments
        self._num_observations = config["env"]["numObservations"]
        self._num_states = config["env"].get("numStates", 0)
        self._num_actions = config["env"]["numActions"]

        self.control_freq_inv = config["env"].get("control_frequency_inv", 1)

        self.clip_obs = config["env"].get("clip_observations", np.Inf)
        self.clip_actions = config["env"].get("clip_actions", np.Inf)

        # set gym spaces for the environment
        self._obs_space = spaces.Box(np.full(self.num_obs, -self.clip_obs),
                                     np.full(self.num_obs, self.clip_obs))
        self._state_space = spaces.Box(np.full(self.num_states, -self.clip_obs),
                                       np.full(self.num_states, self.clip_obs))
        self._act_space = spaces.Box(np.full(self.num_actions, -self.clip_actions),
                                     np.full(self.num_actions, self.clip_actions))

        # define variables to store ranges for observations, states, and action
        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._states_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

    @abc.abstractmethod
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self._obs_space

    @property
    def state_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self._state_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self._act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self._num_environments

    @property
    def num_agents(self) -> int:
        """Get the number of environments."""
        return self._num_agents

    @property
    def num_actions(self) -> int:
        """Get the number of actions in the environment."""
        return self._num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self._num_observations

    @property
    def num_states(self) -> int:
        """Get the number of observations in the environment."""
        return self._num_states


class VecTask(Env):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        super().__init__(config, sim_device, graphics_device_id, headless)

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # Add an instance variable to track the last gravity change time
        self.last_gravity_change_time = time.time()

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "RESET")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_G, "RESET_GOALS")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        # allocate memory to ranges for spaces
        self._states_scale.low = torch.full((self.num_states,), -float('inf'), dtype=torch.float, device=self.device)
        self._states_scale.high = torch.full((self.num_states,), float('inf'), dtype=torch.float, device=self.device)
        self._observations_scale.low = torch.full((self.num_obs,), -float('inf'), dtype=torch.float, device=self.device)
        self._observations_scale.high = torch.full((self.num_obs,), float('inf'), dtype=torch.float, device=self.device)
        self._action_scale.low = torch.full((self.num_actions,), -float('inf'), dtype=torch.float, device=self.device)
        self._action_scale.high = torch.full((self.num_actions,), float('inf'), dtype=torch.float, device=self.device)

        # buffers for filling
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.action_buf = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.prev_action_buf = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def set_sim_params_up_axis(self, sim_params: gymapi.SimParams, axis: str) -> int:
        """Set gravity based on up axis and return axis index.

        Args:
            sim_params: sim params to modify the axis for.
            axis: axis to set sim params for.
        Returns:
            axis index for up axis.
        """
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1
    
    def update_gravity(self):
        current_time = time.time()
        time_since_last_change = current_time - self.last_gravity_change_time

        # Check if it's time to change gravity
        if time_since_last_change >= self.gravity_change_interval:
            # Update gravity here based on your desired logic
            # For example, you can change it randomly or based on some conditions
            # Calculate the new value for x, increasing it by 0.01 each time
            # self.x += 0.0001
            
            # Calculate the new range for random gravity values
            # min_gravity = max(-9.81, 0 - self.x)
            # max_gravity = min(9.81, 0 + self.x)
            # old_gravity=self.sim_params.gravity
            # Here's an example of changing gravity to a random value between -9.81 and 9.81 in the z-axis:
            # new_gravity = gymapi.Vec3(0.0, 0.0, np.random.uniform(min_gravity, max_gravity))
            new_gravity = gymapi.Vec3(np.random.uniform(-0.001, 0.001), 0.0, -9.81)
            # new_gravity = gymapi.Vec3(np.random.uniform(-10, 10), 0.0, -9.81)

            # new_gravity = gymapi.Vec3(0.0, 0.0, np.random.uniform(-9.81, 9.81))

            # Set the new gravity
            # print("before gravity" , self.sim_params.gravity)
            self.sim_params.gravity=new_gravity
            self.gravity =new_gravity
            self.config['sim']['gravity'] =new_gravity

            # Update the last gravity change time
            self.last_gravity_change_time = current_time
            # print("updated gravity" , self.sim_params.gravity)
            

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Apply input action on the environment; reset any which have reached the episode length.

        @note: The returned tensors are on the device set in to the environment config.

        Args:
            action: Action to apply on the simulator. It is a tensor of  shape [N, A] where N is number
                of instances and A is action dimension.

        Returns:
            Tuple containing tensors for observation buffer, rewards, termination state, along with a dictionory
            with extra information.
        """

         # Call the gravity update function
        self.update_gravity()

        self._step_info = {}  # reset the step info

        # if input command is numpy array, convert to torch tensor
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float, device=self.device)

        # check input tensor spec
        action_shape = self.get_action_shape()
        if tuple(action.size()) != action_shape:
            msg = f"Invalid shape for tensor `action`. Input: {tuple(action.size())} != {action_shape}."
            raise ValueError(msg)

        # randomize actions
        action = self.domain_randomizer.apply_action_randomization(action)

        # clip actions to limits
        action = torch.clamp(action, -self.clip_actions, self.clip_actions)

        # apply action
        self.pre_physics_step(action)

        # step physics and render each frame
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # fetch results
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # randomize observations
        self.obs_buf = self.domain_randomizer.apply_observation_randomization(self.obs_buf)

        # fill time out buffer
        if self.max_episode_length is not None:
            self.timeout_buf = torch.where(
                self.progress_buf >= self.max_episode_length - 1,
                torch.ones_like(self.timeout_buf),
                torch.zeros_like(self.timeout_buf)
            )

        # extract the buffers to return
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        rewards = self.rew_buf.to(self.rl_device)
        dones = self.reset_buf.to(self.rl_device)


        self.extras["timeouts"] = self.timeout_buf.to(self.rl_device)

        return self.obs_dict, rewards, dones, self.extras

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        return self.obs_dict

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "RESET" and evt.value > 0:
                    self.reset_buf = torch.ones(size=(self.num_envs,), dtype=torch.long, device=self.device)
                    self.apply_resets()
                elif evt.action == "RESET_GOALS" and evt.value > 0:
                    self.reset_goal_buf = torch.ones(size=(self.num_envs,), dtype=torch.long, device=self.device)
                    self.apply_resets()

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        # sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -50)

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Properties
    """

    def get_gravity(self) -> np.ndarray:
        """Returns the gravity set in the simulator.
        """
        gravity = self.sim_params.gravity
        return np.asarray([gravity.x, gravity.y, gravity.z])

    def get_sim_params(self) -> gymapi.SimParams:
        """Returns the simulator physics parameters.
        """
        return self.sim_params

    def get_state_shape(self) -> torch.Size:
        """Returns the size of the state buffer: [num. instances, num. of state]
        """
        return self.states_buf.size()

    def get_obs_shape(self) -> torch.Size:
        """Returns the size of the observation buffer: [num. instances, num. of obs]
        """
        return self.obs_buf.size()

    def get_action_shape(self) -> torch.Size:
        """Returns the size of the action buffer: [num. instances, num. of action]
        """
        return self.action_buf.size()
