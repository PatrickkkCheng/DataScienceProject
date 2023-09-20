from typing import Deque
import os
import numpy as np
import torch
from collections import deque

from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import quat_unit
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil


from smg_gym.tasks.base_vec_task import VecTask
from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.utils.torch_jit_utils import saturate
from smg_gym.utils.torch_jit_utils import scale_transform

from smg_gym.utils.draw_utils import get_sphere_geom
from smg_gym.assets import add_assets_path

from smg_gym.tasks.smg_object_task_params import SMGObjectTaskDimensions
from smg_gym.tasks.smg_object_task_params import object_properties
from smg_gym.tasks.smg_object_task_params import robot_limits
from smg_gym.tasks.smg_object_task_params import object_limits
from smg_gym.tasks.smg_object_task_params import goal_limits
from smg_gym.tasks.smg_object_task_params import robot_dof_gains
from smg_gym.tasks.smg_object_task_params import robot_dof_properties
from smg_gym.tasks.smg_object_task_params import control_joint_names

from smg_gym.tasks.domain_randomisation import DomainRandomizer


class BaseShadowModularGrasper(VecTask):

    # dimensions useful for SMG hand + object tasks
    _dims = SMGObjectTaskDimensions

    # properties of objects
    _object_properties = object_properties

    # limits of the robot (mapped later: str -> torch.tensor)
    _robot_limits = robot_limits

    # limits of the object (mapped later: str -> torch.tensor)
    _object_limits = object_limits

    # limits of the target (mapped later: str -> torch.tensor)
    _goal_limits = goal_limits

    # PD gains for the robot (mapped later: str -> torch.tensor)
    _robot_dof_gains = robot_dof_gains

    # limits and friction
    _robot_dof_properties = robot_dof_properties

    # actuated joints of the hand
    _control_joint_names = control_joint_names

    # History of state: Number of timesteps to save history for.
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _state_history_len = 2

    # Hand joint states list([num. of instances, num. of dofs])
    _hand_joint_pos_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    _hand_joint_vel_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)

    # Fingertip tcp state list([num. of instances, num. of fingers, 13]) where 13: (pos, quat, linvel, angvel)
    _fingertip_tcp_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)

    # Object root state [num. of instances, 13] where 13: (pos, quat, linvel, angvel)
    _object_base_pos_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    _object_base_orn_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):

        # setup params
        self.cfg = cfg
        self.debug_viz = cfg["env"]["enable_debug_vis"]
        self.env_spacing = cfg["env"]["env_spacing"]
        self.obj_name = cfg["env"]["obj_name"]

        # termination vars
        self.max_episode_length = cfg["env"]["episode_length"]

        # contact sensing
        self.enable_dof_force_sensors = cfg["env"]["enable_dof_force_sensors"]
        self.contact_sensor_modality = cfg["env"]["contact_sensor_modality"]

        # action params
        self.use_sim_pd_control = cfg["env"]["use_sim_pd_control"]
        self.actions_scale = cfg["env"]["actions_scale"]
        self.actions_ema = cfg["env"]["actions_ema"]

        # shared task randomisation params
        self.randomize = cfg["rand_params"]["randomize"]
        self.rand_hand_joints = cfg["rand_params"]["rand_hand_joints"]
        self.rand_obj_init_orn = cfg["rand_params"]["rand_obj_init_orn"]
        self.rand_obj_scale = cfg["rand_params"]["rand_obj_scale"]

        super().__init__(config=cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # change viewer camera
        if self.viewer is not None:
            # cam_pos = gymapi.Vec3(2, 2, 2)
            # cam_target = gymapi.Vec3(0, 0, 1)
            # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            cam_pos = gymapi.Vec3(0.5, 0.5, 0.75)
            cam_target = gymapi.Vec3(0, 0, 0.25)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # initialize the buffers
        self._setup_tensors()

        # spaces for normalising observations
        self._setup_observation_normalisation()

        # inialize buffers that store history of observations
        self.initialize_state_history_buffers()

        # refresh all tensors
        self.refresh_tensors()

    def _setup_tensors(self):
        """
        Allocate memory to various buffers.
        """
        # change constant buffers from numpy/lists into torch tensors
        # limits for robot
        for limit_name in self._robot_limits:
            # extract limit simple-namespace
            limit_dict = self._robot_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        for limit_name in self._goal_limits:
            # extract limit simple-namespace
            limit_dict = self._goal_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)

        # PD gains for actuation
        for gain_name, value in self._robot_dof_gains.items():
            self._robot_dof_gains[gain_name] = torch.tensor(value, dtype=torch.float, device=self.device)

        # get state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.enable_dof_force_sensors:
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        if self.contact_sensor_modality == 'ft_sensor':
            force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # get useful numbers
        self.n_sim_bodies = self.gym.get_sim_rigid_body_count(self.sim)
        self.n_env_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        # create views of actor_root tensor
        # shape = (num_environments, num_actors * 13)
        # 13 -> position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # create views of dof tensor
        # Shape = (num_environments * num_hand_dof, 2)
        # 2 -> position([0]), velocity([1])
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 1]

        # create views of rigid body states
        # shape = (num_environments, num_bodies * 13)
        # 13 -> position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, self.n_env_bodies, 13)

        # create views of contact_force tensor. Obtains the contact force for every rigid body
        # default shape = (n_envs, n_bodies * 3)
        self.contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.n_env_bodies, 3)

        if self.contact_sensor_modality == 'ft_sensor':
            self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(
                self.num_envs, self._dims.NumFingers.value, self._dims.WrenchDim.value
            )
        else:
            self.force_sensor_tensor = torch.zeros(
                size=(self.num_envs, self._dims.NumFingers.value, self._dims.WrenchDim.value)
            )

        if self.enable_dof_force_sensors:
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self._dims.JointTorqueDim.value
            )
        else:
            self.dof_force_tensor = torch.zeros(
                size=(self.num_envs, self._dims.JointTorqueDim.value)
            )

        # setup useful incices
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # setup goal / successes buffers
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

    def _setup_observation_normalisation(self):
        """
        Configures the observations, state and action spaces.
        """

        # Note: This is order sensitive.
        # observations should be appended in the same order as defined in the config
        def get_scale_limits(cfg):
            scale_low, scale_high = [], []
            for key in cfg:
                if cfg[key]:
                    if key in self._robot_limits.keys():
                        scale_low.append(self._robot_limits[key].low)
                        scale_high.append(self._robot_limits[key].high)
                    elif key in self._object_limits.keys():
                        scale_low.append(self._object_limits[key].low)
                        scale_high.append(self._object_limits[key].high)
                    elif key in self._goal_limits.keys():
                        scale_low.append(self._goal_limits[key].low)
                        scale_high.append(self._goal_limits[key].high)
            return scale_low, scale_high

        obs_scale_low, obs_scale_high = get_scale_limits(self.cfg["enabled_obs"])
        self._observations_scale.low = torch.cat(obs_scale_low)
        self._observations_scale.high = torch.cat(obs_scale_high)

        if self.cfg["asymmetric_obs"]:
            state_scale_low, state_scale_high = get_scale_limits(self.cfg["enabled_states"])
            self._states_scale.low = torch.cat(state_scale_low)
            self._states_scale.high = torch.cat(state_scale_high)

        # check that dimensions match observations
        if self._observations_scale.low.shape[0] != self.num_obs or self._observations_scale.high.shape[0] != self.num_obs:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {self.num_obs}."
            raise AssertionError(msg)

        if self.cfg["asymmetric_obs"]:
            if self._states_scale.low.shape[0] != self.num_states or self._states_scale.high.shape[0] != self.num_states:
                msg = f"States scaling dimensions mismatch. " \
                      f"\tLow: {self._states_scale.low.shape[0]}, " \
                      f"\tHigh: {self._states_scale.high.shape[0]}, " \
                      f"\tExpected: {self.num_states}."
                raise AssertionError(msg)

    def initialize_state_history_buffers(self):
        for _ in range(self._state_history_len):
            self._hand_joint_pos_history.append(
                torch.zeros(size=(self.num_envs, self.n_hand_dofs), dtype=torch.float32, device=self.device)
            )
            self._hand_joint_vel_history.append(
                torch.zeros(size=(self.num_envs, self.n_hand_dofs), dtype=torch.float32, device=self.device)
            )
            self._object_base_pos_history.append(
                torch.zeros(size=(self.num_envs, self._dims.PosDim.value), dtype=torch.float32, device=self.device)
            )
            self._object_base_orn_history.append(
                torch.zeros(size=(self.num_envs, self._dims.OrnDim.value), dtype=torch.float32, device=self.device)
            )

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

        # setup domain randomisation
        self.apply_dr = self.cfg['domain_randomization']['randomize']
        self.dr_params = self.cfg['domain_randomization']['dr_params']
        self.domain_randomizer = DomainRandomizer(
            self.sim,
            self.gym,
            self.envs,
            self.apply_dr,
            self.dr_params,
            self.num_envs
        )

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.apply_dr:
            self.domain_randomizer.apply_domain_randomization(
                randomize_buf=None,
                reset_buf=None,
                sim_initialized=self.sim_initialized
            )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.025
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0

        self.gym.add_ground(self.sim, plane_params)

    def _setup_hand(self):

        asset_root = add_assets_path("robot_assets/smg_minitip")
        asset_file = "smg_tactip.urdf"
        # asset_file = "smg_tactip_with_tip_targets.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.collapse_fixed_joints = False
        asset_options.thickness = 0.001
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.flip_visual_attachments = False
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.vhacd_enabled = False
        asset_options.armature = 0.00001
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        self.hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set default hand link properties
        hand_props = self.gym.get_asset_rigid_shape_properties(self.hand_asset)
        for p in hand_props:
            p.friction = 2000.0
            p.torsion_friction = 1000.0
            p.rolling_friction = 0.0
            p.restitution = 0.0
            # p.thickness = 0.001

        # # TODO: remove hardcoded tip indices (be careful of merged fixed links)
        # # set the tip dynamics
        # tip_shape_indices = [7, 14, 21, 28]
        # # tip_shape_indices = [8, 16, 24, 32]
        # for idx in tip_shape_indices:
        #     p = hand_props[idx]
        #     p.friction = 2.0
        #     p.torsion_friction = 1.0
        #     p.rolling_friction = 0.0
        #     p.restitution = 0.0
        #     # p.thickness = 0.001
        # self.gym.set_asset_rigid_shape_properties(self.hand_asset, hand_props)

        self.control_joint_dof_indices = [
            self.gym.find_asset_dof_index(self.hand_asset, name) for name in self._control_joint_names
        ]
        self.control_joint_dof_indices = to_torch(self.control_joint_dof_indices, dtype=torch.long, device=self.device)

        # get counts from hand asset
        self.n_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.n_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        self.n_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)

        # target tensor for updating
        self.target_dof_pos = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.target_dof_vel = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.target_dof_eff = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)

        # init tensor for reference
        self.init_dof_pos = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.init_dof_vel = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.init_dof_eff = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)

        # load saved grasps from cache
        if self.cfg['env']['use_cached_grasps']:
            save_file = os.path.join(
                os.path.dirname(__file__),
                '../saved_grasps/',
                f'{self.obj_name}_grasps.npy'
            )
            self.loaded_grasping_states = torch.from_numpy(np.load(
                save_file
            )).float().to(self.device)

    def _choose_env_object(self):

        if self.obj_name == 'rand':
            self.env_obj_choice = np.random.choice(['sphere', 'box', 'capsule'])
        else:
            self.env_obj_choice = self.obj_name

        if self.env_obj_choice == 'sphere':
            self.env_obj_color = gymapi.Vec3(1, 0, 0)
            if self.rand_obj_scale:
                self.sphere_radius = np.random.uniform(
                    self._object_properties[self.env_obj_choice]['radius_llim'],
                    self._object_properties[self.env_obj_choice]['radius_ulim']
                )
            else:
                self.sphere_radius = self._object_properties[self.env_obj_choice]['radius']

        elif self.env_obj_choice == 'box':
            self.env_obj_color = gymapi.Vec3(0, 0, 1)
            if self.rand_obj_scale:
                self.box_size = np.random.uniform(
                    self._object_properties[self.env_obj_choice]['size_llims'],
                    self._object_properties[self.env_obj_choice]['size_ulims']
                )
            else:
                self.box_size = self._object_properties[self.env_obj_choice]['size']

        elif self.env_obj_choice == 'capsule':
            self.env_obj_color = gymapi.Vec3(0, 1, 0)
            if self.rand_obj_scale:
                self.capsule_radius = np.random.uniform(
                    self._object_properties[self.env_obj_choice]['radius_llim'],
                    self._object_properties[self.env_obj_choice]['radius_ulim']
                )
                self.capsule_width = np.random.uniform(
                    self._object_properties[self.env_obj_choice]['width_llim'],
                    self._object_properties[self.env_obj_choice]['width_ulim']
                )
            else:
                self.capsule_radius = self._object_properties[self.env_obj_choice]['radius']
                self.capsule_width = self._object_properties[self.env_obj_choice]['width']

        else:
            msg = f"Invalid object specified. Input: {self.obj_name} not in ['sphere', 'box', 'rand', 'capsule']."
            raise ValueError(msg)

    def _setup_obj(self):

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_linear_velocity = 10.0
        asset_options.max_angular_velocity = 5.0
        asset_options.thickness = 0.001
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = False

        if self.env_obj_choice == 'sphere':
            self.obj_asset = self.gym.create_sphere(
                self.sim,
                self.sphere_radius,
                asset_options
            )
        elif self.env_obj_choice == 'box':
            self.obj_asset = self.gym.create_box(
                self.sim,
                self.box_size[0],
                self.box_size[1],
                self.box_size[2],
                asset_options
            )
        elif self.env_obj_choice == 'capsule':
            self.obj_asset = self.gym.create_capsule(
                self.sim,
                self.capsule_radius,
                self.capsule_width,
                asset_options
            )

        # set object properties
        obj_props = self.gym.get_asset_rigid_shape_properties(self.obj_asset)
        for p in obj_props:
            p.friction = 2000.0
            p.torsion_friction = 1000.0
            p.rolling_friction = 0.0
            p.restitution = 0.0
            # p.thickness = 0.001
        self.gym.set_asset_rigid_shape_properties(self.obj_asset, obj_props)

        # set initial state for the object
        # self.default_obj_pos = (0.0, -0.016, 0.25)
        # self.default_obj_pos = (0.0, -0.0, 0.245)
        self.default_obj_pos = (0.0, -0.0, 0.235)

        if self.obj_name == 'capsule':
            self.default_obj_orn = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
        else:
            self.default_obj_orn = (0.0, 0.0, 0.0, 1.0)

        self.default_obj_linvel = (0.0, 0.0, 0.0)
        self.default_obj_angvel = (0.0, 0.0, 0.1)
        self.obj_displacement_tensor = to_torch(self.default_obj_pos, dtype=torch.float, device=self.device)

    def _setup_goal(self):
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True

        # asset_root = add_assets_path("object_assets")
        # asset_file = f"{self.obj_name}.urdf"
        # self.goal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        if self.env_obj_choice == 'sphere':
            self.goal_asset = self.gym.create_sphere(
                self.sim,
                self.sphere_radius,
                asset_options
            )
        elif self.env_obj_choice == 'box':
            self.goal_asset = self.gym.create_box(
                self.sim,
                self.box_size[0],
                self.box_size[1],
                self.box_size[2],
                asset_options
            )
        elif self.env_obj_choice == 'capsule':
            self.goal_asset = self.gym.create_capsule(
                self.sim,
                self.capsule_radius,
                self.capsule_width,
                asset_options
            )

        # set initial state of goal
        self.default_goal_pos = (-0.2, -0.06, 0.4)

        if self.obj_name == 'capsule':
            self.default_goal_orn = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
        else:
            self.default_goal_orn = (0.0, 0.0, 0.0, 1.0)

        self.goal_displacement_tensor = to_torch(self.default_goal_pos, dtype=torch.float, device=self.device)

    def _setup_keypoints(self):

        self.kp_dist = 0.06
        self.n_keypoints = self._dims.NumKeypoints.value

        self.obj_kp_positions = torch.zeros(size=(self.num_envs, self.n_keypoints, 3), device=self.device)
        self.goal_kp_positions = torch.zeros(size=(self.num_envs, self.n_keypoints, 3), device=self.device)

        self.kp_basis_vecs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], device=self.device)

        self.kp_geoms = []
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 0, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 1, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 0, 1)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 1, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 1, 1)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 0, 1)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create assets and variables that are fixed per env
        self._setup_hand()
        self._setup_keypoints()

        # get indices useful for tracking contacts and fingertip positions
        self._setup_fingertip_tracking()
        self._setup_contact_tracking()

        # collect useful indeces and handles
        self.envs = []
        self.hand_actor_handles = []
        self.hand_indices = []
        self.obj_actor_handles = []
        self.obj_indices = []
        self.goal_actor_handles = []
        self.goal_indices = []

        for i in range(self.num_envs):

            # create assets and variables that change per env
            self._choose_env_object()
            self._setup_obj()
            self._setup_goal()

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # setup hand
            hand_actor_handle = self._create_hand_actor(env_ptr)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor_handle, gymapi.DOMAIN_SIM)

            # setup obj
            obj_actor_handle = self._create_obj_actor(env_ptr)
            obj_idx = self.gym.get_actor_index(env_ptr, obj_actor_handle, gymapi.DOMAIN_SIM)

            # setup goal
            goal_actor_handle = self._create_goal_actor(env_ptr)
            goal_idx = self.gym.get_actor_index(env_ptr, goal_actor_handle, gymapi.DOMAIN_SIM)

            # append handles and indeces
            self.envs.append(env_ptr)
            self.hand_actor_handles.append(hand_actor_handle)
            self.hand_indices.append(hand_idx)
            self.obj_actor_handles.append(obj_actor_handle)
            self.obj_indices.append(obj_idx)
            self.goal_actor_handles.append(goal_actor_handle)
            self.goal_indices.append(goal_idx)

        # convert indices to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(self.goal_indices, dtype=torch.long, device=self.device)

    def _create_hand_actor(self, env_ptr):

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        self.gym.begin_aggregate(env_ptr, self.n_hand_bodies, self.n_hand_shapes, False)

        handle = self.gym.create_actor(
            env_ptr,
            self.hand_asset,
            pose,
            "hand",
            -1,
            -1
        )

        # enable joint force sensors
        if self.enable_dof_force_sensors:
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

        # Configure DOF properties
        hand_dof_props = self.gym.get_actor_dof_properties(env_ptr, handle)

        # set dof properites based on the control mode
        for dof_index in range(self.n_hand_dofs):

            # Use Isaacgym PD control
            if self.use_sim_pd_control:

                if self.cfg["env"]["command_mode"] == 'position':
                    hand_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_POS
                elif self.cfg["env"]["command_mode"] == 'velocity':
                    hand_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_VEL
                elif self.cfg["env"]["command_mode"] == 'effort':
                    hand_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT

                hand_dof_props['stiffness'][dof_index] = float(self._robot_dof_gains["p_gains"][dof_index])
                hand_dof_props['damping'][dof_index] = float(self._robot_dof_gains["d_gains"][dof_index])

            # Manually compute and apply torque even in position mode
            # (as done in Trifinger paper).
            else:
                hand_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
                hand_dof_props['stiffness'][dof_index] = 0.0
                hand_dof_props['damping'][dof_index] = 0.0

            hand_dof_props['effort'][dof_index] = self._robot_dof_properties["max_torque_Nm"]
            hand_dof_props['velocity'][dof_index] = self._robot_dof_properties["max_velocity_radps"]
            hand_dof_props['friction'][dof_index] = self._robot_dof_properties["friction"][dof_index]
            hand_dof_props['lower'][dof_index] = float(self._robot_limits["joint_pos"].low[dof_index])
            hand_dof_props['upper'][dof_index] = float(self._robot_limits["joint_pos"].high[dof_index])

        self.gym.set_actor_dof_properties(env_ptr, handle, hand_dof_props)

        self.gym.end_aggregate(env_ptr)

        return handle

    def _create_obj_actor(self, env_ptr):

        init_obj_pose = gymapi.Transform()
        init_obj_pose.p = gymapi.Vec3(*self.default_obj_pos)
        init_obj_pose.r = gymapi.Quat(*self.default_obj_orn)

        handle = self.gym.create_actor(
            env_ptr,
            self.obj_asset,
            init_obj_pose,
            "object",
            -1,
            -1
        )

        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        for p in obj_props:
            # p.mass = 0.25
            p.mass = 0.05
            p.inertia.x = gymapi.Vec3(0.001, 0.0, 0.0)
            p.inertia.y = gymapi.Vec3(0.0, 0.001, 0.0)
            p.inertia.z = gymapi.Vec3(0.0, 0.0, 0.001)
        self.gym.set_actor_rigid_body_properties(env_ptr, handle, obj_props)

        # set color of object
        self.gym.set_rigid_body_color(
            env_ptr,
            handle,
            0,
            gymapi.MESH_VISUAL,
            self.env_obj_color
        )

        obj_body_name = self.gym.get_actor_rigid_body_names(env_ptr, handle)
        self.obj_body_idx = self.gym.find_actor_rigid_body_index(env_ptr, handle, obj_body_name[0], gymapi.DOMAIN_ENV)

        return handle

    def _create_goal_actor(self, env_ptr):
        init_goal_pose = gymapi.Transform()
        init_goal_pose.p = gymapi.Vec3(*self.default_goal_pos)
        init_goal_pose.r = gymapi.Quat(*self.default_goal_orn)
        handle = self.gym.create_actor(
            env_ptr,
            self.goal_asset,
            init_goal_pose,
            "goal",
            0,
            0
        )
        # set color of object
        self.gym.set_rigid_body_color(
            env_ptr,
            handle,
            0,
            gymapi.MESH_VISUAL,
            self.env_obj_color
        )
        return handle

    def _setup_contact_tracking(self):

        hand_body_names = self.gym.get_asset_rigid_body_names(self.hand_asset)
        tip_body_names = [name for name in hand_body_names if "tactip_tip" in name]
        non_tip_body_names = [name for name in hand_body_names if "tactip_tip" not in name]
        # tip_body_names = [name for name in hand_body_names if "target" in name]
        # non_tip_body_names = [name for name in hand_body_names if "target" not in name]
        self.tip_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in tip_body_names
        ]
        self.non_tip_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in non_tip_body_names
        ]
        self.n_tips = self._dims.NumFingers.value
        self.n_non_tip_links = len(self.non_tip_body_idxs)

        # add ft sensors to fingertips
        if self.contact_sensor_modality == 'ft_sensor':
            sensor_pose = gymapi.Transform()

            for idx in self.tip_body_idxs:
                sensor_options = gymapi.ForceSensorProperties()
                sensor_options.enable_forward_dynamics_forces = False  # for example gravity
                sensor_options.enable_constraint_solver_forces = True  # for example contacts
                sensor_options.use_world_frame = True
                self.gym.create_asset_force_sensor(self.hand_asset, idx, sensor_pose, sensor_options)

        # rich contacts
        self.contact_positions = torch.zeros(
                (self.num_envs, self.n_tips, self._dims.PosDim.value), dtype=torch.float, device=self.device)
        self.contact_normals = torch.zeros(
                (self.num_envs, self.n_tips, self._dims.VecDim.value), dtype=torch.float, device=self.device)
        self.contact_force_mags = torch.zeros(
                (self.num_envs, self.n_tips, 1), dtype=torch.float, device=self.device)

        self.contact_geom = get_sphere_geom(rad=0.0025, color=(1, 1, 1))

    def _setup_fingertip_tracking(self):

        # hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        hand_body_names = self.gym.get_asset_rigid_body_names(self.hand_asset)
        tcp_body_names = [name for name in hand_body_names if "tcp" in name]
        self.fingertip_tcp_body_idxs = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in tcp_body_names
        ]

    def get_fingertip_contacts(self):

        # get envs where obj is contacted
        bool_obj_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.obj_body_idx, :], dim=1) > 0,
            torch.ones(size=(self.num_envs,), device=self.device),
            torch.zeros(size=(self.num_envs,), device=self.device),
        )

        # get envs where tips are contacted
        net_tip_contact_forces = self.contact_force_tensor[:, self.tip_body_idxs, :]
        bool_tip_contacts = torch.where(
            torch.count_nonzero(net_tip_contact_forces, dim=2) > 0,
            torch.ones(size=(self.num_envs, self.n_tips), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device),
        )

        # get all the contacted links that are not the tip
        net_non_tip_contact_forces = self.contact_force_tensor[:, self.non_tip_body_idxs, :]
        bool_non_tip_contacts = torch.where(
            torch.count_nonzero(net_non_tip_contact_forces, dim=2) > 0,
            torch.ones(size=(self.num_envs, self.n_non_tip_links), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_non_tip_links), device=self.device),
        )
        n_non_tip_contacts = torch.sum(bool_non_tip_contacts, dim=1)

        # repeat for n_tips shape=(n_envs, n_tips)
        onehot_obj_contacts = bool_obj_contacts.unsqueeze(1).repeat(1, self.n_tips)

        # get envs where object and tips are contacted
        tip_object_contacts = torch.where(
            onehot_obj_contacts > 0,
            bool_tip_contacts,
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device)
        )
        n_tip_contacts = torch.sum(bool_tip_contacts, dim=1)

        return net_tip_contact_forces, tip_object_contacts, n_tip_contacts, n_non_tip_contacts

    def get_rich_fingertip_contacts(self):
        """
        Contact Properties
            'env0', 'env1',
            'body0', 'body1',
            'localPos0', 'localPos1',
            'minDist',
            'initialOverlap',
            'normal',
            'offset0', 'offset1',
            'lambda',
            'lambdaFriction',
            'friction',
            'torsionFriction',
            'rollingFriction'
        """
        if self.device != 'cpu':
            raise ValueError("Rich contacts not available with GPU pipeline.")

        # iterate through environment to pull all contact info
        for i in range(self.num_envs):
            contacts = self.gym.get_env_rigid_contacts(self.envs[i])

            # print('')
            # print(contacts['lambda'])
            # print(contacts['friction'])
            # print(contacts['lambdaFriction'])
            # print(contacts['torsionFriction'])
            # print(contacts['rollingFriction'])

            # accumulate all contacts within an environment
            for j, tip_body_idx in enumerate(self.tip_body_idxs):
                tip_contacts = contacts[np.where(
                    (contacts['body0'] == self.obj_body_idx)
                    & (contacts['body1'] == tip_body_idx)
                )]
                self.contact_positions[i, j, :] = to_torch([
                    tip_contacts['localPos1']['x'].mean() if len(tip_contacts['localPos1']['x']) > 0 else 0.0,
                    tip_contacts['localPos1']['y'].mean() if len(tip_contacts['localPos1']['y']) > 0 else 0.0,
                    tip_contacts['localPos1']['z'].mean() if len(tip_contacts['localPos1']['z']) > 0 else 0.0
                ], device=self.device)
                self.contact_normals[i, j, :] = to_torch([
                    tip_contacts['normal']['x'].mean() if len(tip_contacts['normal']['x']) > 0 else 0.0,
                    tip_contacts['normal']['y'].mean() if len(tip_contacts['normal']['y']) > 0 else 0.0,
                    tip_contacts['normal']['z'].mean() if len(tip_contacts['normal']['z']) > 0 else 0.0
                ], device=self.device)
                self.contact_force_mags[i, j, :] = to_torch([
                    tip_contacts['lambda'].mean() if len(tip_contacts['lambda']) > 0 else 0.0
                ], device=self.device)

    def pre_physics_step(self, actions):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        self.apply_resets()
        self.apply_actions(actions)

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

        self.progress_buf += 1
        self.randomize_buf += 1

        self.refresh_tensors()
        self.compute_observations()
        self.fill_observation_buffer()
        self.fill_states_buffer()
        self.norm_obs_state_buffer()
        self.compute_reward_and_termination()

        if self.viewer and self.debug_viz:
            self.visualise_features()

    def refresh_tensors(self):
        """
        Refresh all state tensors.
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.contact_sensor_modality == 'ft_sensor':
            self.gym.refresh_force_sensor_tensor(self.sim)

        if self.enable_dof_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

    def standard_fill_buffer(self, buf, buf_cfg):
        """
        Fill observation buffer with observations shared across different tasks.
        """

        start_offset, end_offset = 0, 0

        # joint position
        if buf_cfg["joint_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointPositionDim.value
            buf[:, start_offset:end_offset] = self.hand_joint_pos

        # joint velocity
        if buf_cfg["joint_vel"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointVelocityDim.value
            buf[:, start_offset:end_offset] = self.hand_joint_vel

        # joint dof force
        if buf_cfg["joint_eff"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointTorqueDim.value
            buf[:, start_offset:end_offset] = self.dof_force_tensor

        # fingertip positions
        if buf_cfg["fingertip_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingertipPosDim.value
            buf[:, start_offset:end_offset] = self.fingertip_pos

        # fingertip orn
        if buf_cfg["fingertip_orn"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingertipOrnDim.value
            buf[:, start_offset:end_offset] = self.fingertip_orn.reshape(
                self.num_envs, self._dims.FingertipOrnDim.value)

        # latest actions
        if buf_cfg["latest_action"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.ActionDim.value
            buf[:, start_offset:end_offset] = self.action_buf

        # previous actions
        if buf_cfg["prev_action"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.ActionDim.value
            buf[:, start_offset:end_offset] = self.prev_action_buf

        # target joint position
        if buf_cfg["target_joint_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointPositionDim.value
            buf[:, start_offset:end_offset] = self.target_dof_pos

        # boolean tips in contacts
        if buf_cfg["bool_tip_contacts"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.NumFingers.value
            buf[:, start_offset:end_offset] = self.tip_object_contacts

        # tip contact forces
        if buf_cfg["net_tip_contact_forces"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingerContactForceDim.value
            buf[:, start_offset:end_offset] = self.net_tip_contact_forces.reshape(
                self.num_envs, self._dims.FingerContactForceDim.value)

        # tip contact forces (ft sensors enabled)
        if buf_cfg["ft_sensor_contact_forces"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingerContactForceDim.value
            buf[:, start_offset:end_offset] = self.force_sensor_tensor[:, :, 0:3].reshape(
                self.num_envs, self._dims.FingerContactForceDim.value)

        # tip contact torques (ft sensors enabled)
        if buf_cfg["ft_sensor_contact_torques"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingerContactTorqueDim.value
            buf[:, start_offset:end_offset] = self.force_sensor_tensor[:, :, 3:6].reshape(
                self.num_envs, self._dims.FingerContactTorqueDim.value)

        # tip contact positions (rich contacts enabled)
        if buf_cfg["tip_contact_positions"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingertipPosDim.value
            buf[:, start_offset:end_offset] = self.contact_positions.reshape(
                self.num_envs, self._dims.FingertipPosDim.value)

        # tip contact normals (rich contacts enabled)
        if buf_cfg["tip_contact_normals"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.FingerContactForceDim.value
            buf[:, start_offset:end_offset] = self.contact_normals.reshape(
                self.num_envs, self._dims.FingerContactForceDim.value)

        # tip contact force magnitudes (rich contacts enabled)
        if buf_cfg["tip_contact_force_mags"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.NumFingers.value
            buf[:, start_offset:end_offset] = self.contact_force_mags.reshape(
                self.num_envs, self._dims.NumFingers.value)

        # object position
        if buf_cfg["object_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.PosDim.value
            buf[:, start_offset:end_offset] = self.obj_base_pos

        # object orientation
        if buf_cfg["object_orn"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.OrnDim.value
            buf[:, start_offset:end_offset] = self.obj_base_orn

        # object keypoints
        if buf_cfg["object_kps"]:
            rel_kp_pos = (self.obj_kp_positions - self.obj_displacement_tensor).reshape(self.num_envs,
                                                                                        self._dims.KeypointPosDim.value)
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointPosDim.value
            buf[:, start_offset:end_offset] = rel_kp_pos

        # object linear velocity
        if buf_cfg["object_linvel"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.LinearVelocityDim.value
            buf[:, start_offset:end_offset] = self.obj_base_linvel

        # object angular velocity
        if buf_cfg["object_angvel"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.AngularVelocityDim.value
            buf[:, start_offset:end_offset] = self.obj_base_angvel

        # goal position
        if buf_cfg["goal_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.PosDim.value
            buf[:, start_offset:end_offset] = self.goal_base_pos

        # goal orientation
        if buf_cfg["goal_orn"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.OrnDim.value
            buf[:, start_offset:end_offset] = self.goal_base_orn

        # goal keypoints
        if buf_cfg["goal_kps"]:

            rel_goal_pos = (self.goal_kp_positions - self.goal_displacement_tensor).reshape(self.num_envs,
                                                                                            self._dims.KeypointPosDim.value)
            start_offset = end_offset
            end_offset = start_offset + self._dims.KeypointPosDim.value
            buf[:, start_offset:end_offset] = rel_goal_pos

        # active quat between goal and object
        if buf_cfg["active_quat"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.OrnDim.value
            buf[:, start_offset:end_offset] = self.active_quat

        return start_offset, end_offset

    def calculate_buffer_size(self, buf_cfg):
        """
        Calculate size of the buffer for observations and states.
        """

        buf_size = 0

        if "joint_pos" in buf_cfg.keys() and buf_cfg["joint_pos"]:
            buf_size += self._dims.JointPositionDim.value
        if "joint_vel" in buf_cfg.keys() and buf_cfg["joint_vel"]:
            buf_size += self._dims.JointVelocityDim.value
        if "joint_eff" in buf_cfg.keys() and buf_cfg["joint_eff"]:
            buf_size += self._dims.JointTorqueDim.value
        if "fingertip_pos" in buf_cfg.keys() and buf_cfg["fingertip_pos"]:
            buf_size += self._dims.FingertipPosDim.value
        if "fingertip_orn" in buf_cfg.keys() and buf_cfg["fingertip_orn"]:
            buf_size += self._dims.FingertipOrnDim.value
        if "latest_action" in buf_cfg.keys() and buf_cfg["latest_action"]:
            buf_size += self._dims.ActionDim.value
        if "prev_action" in buf_cfg.keys() and buf_cfg["prev_action"]:
            buf_size += self._dims.ActionDim.value
        if "target_joint_pos" in buf_cfg.keys() and buf_cfg["target_joint_pos"]:
            buf_size += self._dims.JointPositionDim.value
        if "bool_tip_contacts" in buf_cfg.keys() and buf_cfg["bool_tip_contacts"]:
            buf_size += self._dims.NumFingers.value
        if "net_tip_contact_forces" in buf_cfg.keys() and buf_cfg["net_tip_contact_forces"]:
            buf_size += self._dims.FingerContactForceDim.value
        if "ft_sensor_contact_forces" in buf_cfg.keys() and buf_cfg["ft_sensor_contact_forces"]:
            buf_size += self._dims.FingerContactForceDim.value
        if "ft_sensor_contact_torques" in buf_cfg.keys() and buf_cfg["ft_sensor_contact_torques"]:
            buf_size += self._dims.FingerContactTorqueDim.value
        if "tip_contact_positions" in buf_cfg.keys() and buf_cfg["tip_contact_positions"]:
            buf_size += self._dims.FingertipPosDim.value
        if "tip_contact_normals" in buf_cfg.keys() and buf_cfg["tip_contact_normals"]:
            buf_size += self._dims.FingerContactForceDim.value
        if "tip_contact_force_mags" in buf_cfg.keys() and buf_cfg["tip_contact_force_mags"]:
            buf_size += self._dims.NumFingers.value
        if "object_pos" in buf_cfg.keys() and buf_cfg["object_pos"]:
            buf_size += self._dims.PosDim.value
        if "object_orn" in buf_cfg.keys() and buf_cfg["object_orn"]:
            buf_size += self._dims.OrnDim.value
        if "object_kps" in buf_cfg.keys() and buf_cfg["object_kps"]:
            buf_size += self._dims.KeypointPosDim.value
        if "object_linvel" in buf_cfg.keys() and buf_cfg["object_linvel"]:
            buf_size += self._dims.LinearVelocityDim.value
        if "object_angvel" in buf_cfg.keys() and buf_cfg["object_angvel"]:
            buf_size += self._dims.AngularVelocityDim.value
        if "goal_pos" in buf_cfg.keys() and buf_cfg["goal_pos"]:
            buf_size += self._dims.PosDim.value
        if "goal_orn" in buf_cfg.keys() and buf_cfg["goal_orn"]:
            buf_size += self._dims.OrnDim.value
        if "goal_kps" in buf_cfg.keys() and buf_cfg["goal_kps"]:
            buf_size += self._dims.KeypointPosDim.value
        if "active_quat" in buf_cfg.keys() and buf_cfg["active_quat"]:
            buf_size += self._dims.OrnDim.value
        if "pivot_axel_vec" in buf_cfg.keys() and buf_cfg["pivot_axel_vec"]:
            buf_size += self._dims.VecDim.value
        if "pivot_axel_pos" in buf_cfg.keys() and buf_cfg["pivot_axel_pos"]:
            buf_size += self._dims.PosDim.value

        return buf_size

    def norm_obs_state_buffer(self):
        """
        Normalise the observation and state buffer
        """
        if self.cfg["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )
            # for asymmetric obs
            if self.cfg["asymmetric_obs"]:
                self.states_buf = scale_transform(
                    self.states_buf,
                    lower=self._states_scale.low,
                    upper=self._states_scale.high
                )

    def reset_hand(self, env_ids_for_reset):
        """
        Reset joint positions on hand, randomisation limits handled in init_joint_mins/maxs.
        """
        num_envs_to_reset = len(env_ids_for_reset)

        if self.cfg["env"]["use_cached_grasps"] and hasattr(self, 'sampled_pose_idx'):
            # randomly select from loaded grasps
            target_dof_pos = self.loaded_grasping_states[self.sampled_pose_idx].clone()[:, :self._dims.JointPositionDim.value]
        else:
            # add randomisation to the joint poses
            if self.randomize and self.rand_hand_joints:
                # sample uniform random from (-1, 1)
                rand_stddev = torch_rand_float(-1.0, 1.0, (num_envs_to_reset, self.n_hand_dofs), device=self.device)

                # add noise to DOF positions
                delta_max = self._robot_limits["joint_pos"].rand_uplim - self._robot_limits["joint_pos"].default
                delta_min = self._robot_limits["joint_pos"].rand_lolim - self._robot_limits["joint_pos"].default
                target_dof_pos = self._robot_limits["joint_pos"].default + \
                    delta_min + (delta_max - delta_min) * rand_stddev
            else:
                target_dof_pos = self._robot_limits["joint_pos"].default

        # get default velocity and effort
        target_dof_vel = self._robot_limits["joint_vel"].default
        target_dof_eff = self._robot_limits["joint_eff"].default

        self.target_dof_pos[env_ids_for_reset, :self.n_hand_dofs] = target_dof_pos
        self.target_dof_vel[env_ids_for_reset, :self.n_hand_dofs] = target_dof_vel
        self.target_dof_eff[env_ids_for_reset, :self.n_hand_dofs] = target_dof_eff

        self.init_dof_pos[env_ids_for_reset, :self.n_hand_dofs] = target_dof_pos
        self.init_dof_vel[env_ids_for_reset, :self.n_hand_dofs] = target_dof_vel
        self.init_dof_eff[env_ids_for_reset, :self.n_hand_dofs] = target_dof_eff

        # reset robot fingertips state history
        for idx in range(1, self._state_history_len):
            self._hand_joint_pos_history[idx][env_ids_for_reset] = 0.0
            self._hand_joint_vel_history[idx][env_ids_for_reset] = 0.0

        # fill first sample from buffer to allow for deltas on next step
        self._hand_joint_pos_history[0][env_ids_for_reset, :] = target_dof_pos
        self._hand_joint_vel_history[0][env_ids_for_reset, :] = target_dof_vel

        # set DOF states to those reset
        hand_ids_int32 = self.hand_indices[env_ids_for_reset].to(torch.int32)

        # reset the targets for the sim pd_controller
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.target_dof_pos),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )
        self.gym.set_dof_velocity_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.target_dof_vel),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.target_dof_eff),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )

        # reset the state of the dofs
        self.dof_pos[env_ids_for_reset, :] = target_dof_pos
        self.dof_vel[env_ids_for_reset, :] = target_dof_vel
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )

    def reset_object(self, env_ids_for_reset):
        """
        Reset the pose of the object.
        """

        num_envs_to_reset = len(env_ids_for_reset)

        if self.cfg["env"]["use_cached_grasps"] and hasattr(self, 'sampled_pose_idx'):
            # randomly select from loaded grasps
            object_pose = self.loaded_grasping_states[self.sampled_pose_idx].clone(
            )[:, self._dims.JointPositionDim.value:self._dims.JointPositionDim.value+self._dims.PoseDim.value]
            object_pos = object_pose[:, 0:3]
            object_orn = object_pose[:, 3:7]
        else:

            # set obj pos and vel to default
            object_pos = to_torch(self.default_obj_pos, dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))

            # randomise object rotation
            if self.randomize and self.rand_obj_init_orn:
                rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids_for_reset), 2), device=self.device)
                object_orn = randomize_rotation(
                    rand_floats[:, 0],
                    rand_floats[:, 1],
                    self.x_unit_tensor[env_ids_for_reset],
                    self.y_unit_tensor[env_ids_for_reset]
                )
            else:
                object_orn = to_torch(self.default_obj_orn, dtype=torch.float,
                                      device=self.device).repeat((num_envs_to_reset, 1))

        object_linvel = to_torch(self.default_obj_linvel, dtype=torch.float,
                                 device=self.device).repeat((num_envs_to_reset, 1))
        object_angvel = to_torch(self.default_obj_angvel, dtype=torch.float,
                                 device=self.device).repeat((num_envs_to_reset, 1))

        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 0:3] = object_pos
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 3:7] = object_orn
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 7:10] = object_linvel
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 10:13] = object_angvel

        # fill first sample from buffer to allow for deltas on next step
        self._object_base_pos_history[0][env_ids_for_reset, :] = object_pos
        self._object_base_orn_history[0][env_ids_for_reset, :] = object_orn

        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_base_pos_history[idx][env_ids_for_reset] = 0.0
            self._object_base_orn_history[idx][env_ids_for_reset] = 0.0

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose of the object.
        """
        pass

    def apply_resets(self):
        """
        Logic for applying resets
        """

        # If randomizing, apply on env resets
        if self.apply_dr:
            self.randomize_buf = self.domain_randomizer.apply_domain_randomization(
                randomize_buf=self.randomize_buf,
                reset_buf=self.reset_buf,
                sim_initialized=self.sim_initialized
            )

        env_ids_for_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids_for_reset = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # randomly select from loaded grasps
        if self.cfg["env"]["use_cached_grasps"]:
            self.sampled_pose_idx = np.random.randint(self.loaded_grasping_states.shape[0], size=len(env_ids_for_reset))

        actor_root_state_reset_indices = []

        # nothing to reset
        if len(env_ids_for_reset) == 0 and len(goal_env_ids_for_reset) == 0:
            return None

        # reset goals
        if len(goal_env_ids_for_reset) > 0:
            self.reset_target_pose(goal_env_ids_for_reset)
            actor_root_state_reset_indices.append(self.goal_indices[goal_env_ids_for_reset])

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_object(env_ids_for_reset)
            self.reset_hand(env_ids_for_reset)
            self.reset_target_pose(env_ids_for_reset)
            actor_root_state_reset_indices.append(self.obj_indices[env_ids_for_reset])
            actor_root_state_reset_indices.append(self.goal_indices[env_ids_for_reset])

        # set the root state tensor to reset object and goal pose
        # has to be done together for some reason...
        reset_indices = torch.unique(
            torch.cat(actor_root_state_reset_indices)
        ).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(reset_indices),
            len(reset_indices)
        )

        # reset buffers
        self.progress_buf[env_ids_for_reset] = 0
        self.action_buf[env_ids_for_reset] = 0
        self.prev_action_buf[env_ids_for_reset] = 0
        self.reset_buf[env_ids_for_reset] = 0
        self.successes[env_ids_for_reset] = 0

    def apply_actions(self, actions):
        """
        Setting of input actions into simulator before performing the physics simulation step.
        Actions are in action_buf variable.
        """
        # zero/positive/negative actions for debugging
        # actions = torch.zeros_like(self.action_buf)
        # actions = torch.ones_like(self.action_buf)
        # actions = torch.ones_like(self.action_buf) * -1

        if self.use_sim_pd_control:
            self.apply_actions_sim_pd(actions)
        else:
            self.apply_actions_custom_pd(actions)

    def apply_actions_sim_pd(self, actions):
        """
        Use IsaacGym PD controller for applying actions.
        """

        if self.cfg["env"]["command_mode"] == 'position':

            # increment actions in buffer with new actions
            scaled_actions = actions.clone().to(self.device) * self.actions_scale
            self.action_buf = self.actions_ema * scaled_actions + \
                (1-self.actions_ema) * self.prev_action_buf
            self.prev_action_buf = self.action_buf.clone()

            # limit to max change per step
            self.action_buf = torch.clamp(
                self.action_buf,
                -self._robot_dof_properties["max_position_delta_rad"],
                self._robot_dof_properties["max_position_delta_rad"]
            )

            # set new target position
            self.target_dof_pos += self.action_buf

            # limit new target position within joint limits
            self.target_dof_pos = saturate(
                self.target_dof_pos,
                lower=self._robot_limits['joint_pos'].low,
                upper=self._robot_limits['joint_pos'].high
            )

            # send target position to sim
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))

        elif self.cfg["env"]["command_mode"] == 'velocity':

            # increment actions in buffer with new actions
            scaled_actions = actions.clone().to(self.device) * self.actions_scale
            self.action_buf = self.actions_ema * scaled_actions + \
                (1-self.actions_ema) * self.prev_action_buf
            self.prev_action_buf = self.action_buf.clone()

            # limit to max change per step
            self.action_buf = torch.clamp(
                self.action_buf,
                -self._robot_dof_properties["max_velocity_radps"],
                self._robot_dof_properties["max_velocity_radps"]
            )

            # set new target velocity
            self.target_dof_vel += self.action_buf

            # limit new target velocity within limits
            self.target_dof_vel = saturate(
                self.target_dof_vel,
                lower=self._robot_limits['joint_vel'].low,
                upper=self._robot_limits['joint_vel'].high
            )

            # send target velocity to sim
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_vel))

        # compute command on the basis of mode selected
        elif self.cfg["env"]["command_mode"] == 'torque':

            # set new target effort
            self.target_dof_eff = actions.clone().to(self.device) * self.actions_scale

            # limit new target velocity within joint limits
            self.target_dof_eff = saturate(
                self.target_dof_eff,
                lower=self._robot_limits["joint_eff"].low,
                upper=self._robot_limits["joint_eff"].high
            )

            # set computed torques to simulator buffer.
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_eff))

        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['position', 'velocity', 'torque']."
            raise ValueError(msg)

    def apply_actions_custom_pd(self, actions):
        """
        Use custom PD controller for applying actions.
        """
        if self.cfg["env"]["command_mode"] == 'position':

            # increment actions in buffer with new actions
            scaled_actions = actions.clone().to(self.device) * self.actions_scale
            self.action_buf = self.actions_ema * scaled_actions + \
                (1-self.actions_ema) * self.prev_action_buf
            self.prev_action_buf = self.action_buf.clone()

            # limit to max change per step
            self.action_buf = torch.clamp(
                self.action_buf,
                -self._robot_dof_properties["max_position_delta_rad"],
                self._robot_dof_properties["max_position_delta_rad"]
            )

            # set new target position
            self.target_dof_pos += self.action_buf

            # limit new target position within joint limits
            self.target_dof_pos = saturate(
                self.target_dof_pos,
                lower=self._robot_limits['joint_pos'].low,
                upper=self._robot_limits['joint_pos'].high
            )

            # compute torque to apply
            # TODO: perhaps change error to shortest angular distance withing limts
            error = self.target_dof_pos - self.dof_pos
            self.target_dof_eff = self._robot_dof_gains["p_gains"] * error
            self.target_dof_eff -= self._robot_dof_gains["d_gains"] * self.dof_vel

        elif self.cfg["env"]["command_mode"] == 'velocity':

            # increment actions in buffer with new actions
            scaled_actions = actions.clone().to(self.device) * self.actions_scale
            self.action_buf = self.actions_ema * scaled_actions + \
                (1-self.actions_ema) * self.prev_action_buf
            self.prev_action_buf = self.action_buf.clone()

            # limit to max change per step
            self.action_buf = torch.clamp(
                self.action_buf,
                -self._robot_dof_properties["max_velocity_radps"],
                self._robot_dof_properties["max_velocity_radps"]
            )

            # set new target velocity
            self.target_dof_vel += self.action_buf

            # limit new target velocity within limits
            self.target_dof_vel = saturate(
                self.target_dof_vel,
                lower=self._robot_limits['joint_vel'].low,
                upper=self._robot_limits['joint_vel'].high
            )

            # compute torque to apply
            error = self.target_dof_vel - self.dof_vel
            self.target_dof_eff = self._robot_dof_gains["d_gains"] * error

        elif self.cfg["env"]["command_mode"] == 'torque':
            # set new target effort
            self.target_dof_eff = actions.clone().to(self.device) * self.actions_scale

        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['position', 'velocity', 'torque']."
            raise ValueError(msg)

        # apply clamping of computed torque to actuator limits
        self.target_dof_eff = saturate(
            self.target_dof_eff,
            lower=self._robot_limits["joint_eff"].low,
            upper=self._robot_limits["joint_eff"].high
        )

        # set computed torques to simulator buffer.
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_eff))

    def canonicalise_quat(self, quat):
        canon_quat = quat_unit(quat)
        canon_quat[torch.where(canon_quat[..., 3] < 0)] *= -1
        return canon_quat

    def compute_observations(self):

        # get which tips are in contact
        if self.contact_sensor_modality == 'rich_cpu':
            self.get_rich_fingertip_contacts()

        (self.net_tip_contact_forces,
         self.tip_object_contacts,
         self.n_tip_contacts,
         self.n_non_tip_contacts) = self.get_fingertip_contacts()

        # get tcp positions
        fingertip_states = self.rigid_body_tensor[:, self.fingertip_tcp_body_idxs, :]
        self.fingertip_pos = fingertip_states[..., 0:3].reshape(self.num_envs, self._dims.FingertipPosDim.value)
        self.fingertip_orn = self.canonicalise_quat(fingertip_states[..., 3:7])
        self.fingertip_linvel = fingertip_states[..., 7:10]
        self.fingertip_angvel = fingertip_states[..., 10:13]

        # get hand joint pos and vel
        self.hand_joint_pos = self.dof_pos[:, :].squeeze()
        self.hand_joint_vel = self.dof_vel[:, :].squeeze()

        # use position deltas instead of velocity tensor
        # self.hand_joint_vel = self._hand_joint_pos_history[0] - self._hand_joint_pos_history[1]
        # self.hand_joint_acc = self._hand_joint_vel_history[0] - self._hand_joint_vel_history[1]

        # get object pose / vel
        self.obj_base_pos = self.root_state_tensor[self.obj_indices, 0:3]
        self.obj_base_orn = self.canonicalise_quat(self.root_state_tensor[self.obj_indices, 3:7])
        self.obj_base_linvel = self.root_state_tensor[self.obj_indices, 7:10]
        self.obj_base_angvel = self.root_state_tensor[self.obj_indices, 10:13]

        # get goal pose
        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.canonicalise_quat(self.root_state_tensor[self.goal_indices, 3:7])

        # relative goal pose w.r.t. obj pose
        self.active_quat = self.canonicalise_quat(quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn)))

        # update the current keypoint positions
        for i in range(self.n_keypoints):
            self.obj_kp_positions[:, i, :] = self.obj_base_pos + \
                quat_rotate(self.obj_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)
            self.goal_kp_positions[:, i, :] = self.goal_base_pos + \
                quat_rotate(self.goal_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)

        # append observations to history stack
        self._hand_joint_pos_history.appendleft(self.hand_joint_pos.clone())
        self._hand_joint_vel_history.appendleft(self.hand_joint_vel.clone())
        self._object_base_pos_history.appendleft(self.obj_base_pos.clone())
        self._object_base_orn_history.appendleft(self.obj_base_orn.clone())

    def visualise_features(self):

        self.gym.clear_lines(self.viewer)

        for i in range(self.num_envs):

            # draw rich contacts
            if self.contact_sensor_modality == 'rich_cpu' and self.device == 'cpu':

                contact_pose = gymapi.Transform()
                contact_pose.r = gymapi.Quat(0, 0, 0, 1)

                rigid_body_poses = self.gym.get_actor_rigid_body_states(
                    self.envs[i],
                    self.hand_actor_handles[i],
                    gymapi.STATE_POS
                )['pose']

                for j in range(self.n_tips):

                    # get contact positions
                    contact_position = gymapi.Vec3(
                        self.contact_positions[i, j, 0],
                        self.contact_positions[i, j, 1],
                        self.contact_positions[i, j, 2]
                    )
                    contact_normal = gymapi.Vec3(
                        self.contact_normals[i, j, 0],
                        self.contact_normals[i, j, 1],
                        self.contact_normals[i, j, 2]
                    )
                    # transform with tip pose
                    tip_pose = gymapi.Transform.from_buffer(rigid_body_poses[self.tip_body_idxs[j]])
                    contact_pose.p = tip_pose.transform_point(contact_position)

                    # draw contact position
                    gymutil.draw_lines(
                        self.contact_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        contact_pose
                    )

                    # draw contact normals
                    gymutil.draw_line(
                        contact_pose.p,
                        contact_pose.p + contact_normal * self.contact_force_mags[i, j],
                        gymapi.Vec3(1.0, 0.0, 0.0),
                        self.gym,
                        self.viewer,
                        self.envs[i],
                    )

            # draw ft sensor contacts
            elif self.contact_sensor_modality == 'ft_sensor' and self.device == 'cpu':

                rigid_body_poses = self.gym.get_actor_rigid_body_states(
                    self.envs[i],
                    self.hand_actor_handles[i],
                    gymapi.STATE_POS
                )['pose']

                for j in range(self.n_tips):
                    tip_pose = gymapi.Transform.from_buffer(rigid_body_poses[self.tip_body_idxs[j]])

                    contact_force = gymapi.Vec3(
                        self.force_sensor_tensor[i, j, 0],
                        self.force_sensor_tensor[i, j, 1],
                        self.force_sensor_tensor[i, j, 2]
                    )

                    gymutil.draw_line(
                        tip_pose.p,
                        tip_pose.p + contact_force,
                        gymapi.Vec3(1.0, 0.0, 0.0),
                        self.gym,
                        self.viewer,
                        self.envs[i],
                    )

            # draw default contacts
            elif self.device == 'cpu':
                contact_color = gymapi.Vec3(1, 0, 0)
                self.gym.draw_env_rigid_contacts(self.viewer, self.envs[i], contact_color, 0.2, True)

            # draw object and goal keypoints
            for j in range(self.n_keypoints):
                pose = gymapi.Transform()

                # visualise object keypoints
                pose.p = gymapi.Vec3(
                    self.obj_kp_positions[i, j, 0],
                    self.obj_kp_positions[i, j, 1],
                    self.obj_kp_positions[i, j, 2]
                )

                pose.r = gymapi.Quat(0, 0, 0, 1)

                gymutil.draw_lines(
                    self.kp_geoms[j],
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    pose
                )

                # visualise goal keypoints
                pose.p = gymapi.Vec3(
                    self.goal_kp_positions[i, j, 0],
                    self.goal_kp_positions[i, j, 1],
                    self.goal_kp_positions[i, j, 2]
                )

                gymutil.draw_lines(
                    self.kp_geoms[j],
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    pose
                )
