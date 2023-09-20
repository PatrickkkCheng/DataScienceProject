"""
Train:
python train.py task=smg_debug headless=True
"""

import torch
from isaacgym.torch_utils import to_torch
from isaacgym import gymapi

from smg_gym.assets import add_assets_path
from smg_gym.tasks.base_hand import BaseShadowModularGrasper


class SMGDebug(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):
        cfg["env"]["numObservations"] = self.calculate_buffer_size(cfg["enabled_obs"])

        if cfg["asymmetric_obs"]:
            cfg["env"]["numStates"] = self.calculate_buffer_size(cfg["enabled_states"])

        cfg["env"]["numActions"] = self._dims.ActionDim.value

        super(SMGDebug, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

        # set default joint positions
        self._robot_limits["joint_pos"].default *= 0.0

    def _setup_obj(self):
        """
        Move obj out of the way and disable collisions/gravity
        """

        asset_root = add_assets_path("object_assets")
        asset_file = f"{self.obj_name}.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        self.obj_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set initial state for the object
        self.default_obj_pos = (-0.2, -0.06, 0.4)
        self.default_obj_orn = (0.0, 0.0, 0.0, 1.0)
        self.default_obj_linvel = (0.0, 0.0, 0.0)
        self.default_obj_angvel = (0.0, 0.0, 0.1)
        self.obj_displacement_tensor = to_torch(self.default_obj_pos, dtype=torch.float, device=self.device)

    def fill_observation_buffer(self):
        """
        Fill observation buffer.
        shape = (num_envs, num_obs)
        """

        # fill obs buffer with observations shared across tasks.
        obs_cfg = self.cfg["enabled_obs"]
        start_offset, end_offset = self.standard_fill_buffer(self.obs_buf, obs_cfg)

        return self.obs_buf

    def fill_states_buffer(self):
        """
        Fill states buffer.
        shape = (num_envs, num_obs)
        """
        # if states spec is empty then return
        if not self.cfg["asymmetric_obs"]:
            return

        # fill obs buffer with observations shared across tasks.
        states_cfg = self.cfg["enabled_states"]
        start_offset, end_offset = self.standard_fill_buffer(self.states_buf, states_cfg)

        return self.states_buf
