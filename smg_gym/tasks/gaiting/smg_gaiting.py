"""
Train:
python train.py task=smg_gaiting headless=True

# train low env num
python train.py task=smg_gaiting task.env.num_envs=8 headless=False train.params.config.horizon_length=1024  train.params.config.minibatch_size=32

Test:
python train.py task=smg_gaiting task.env.num_envs=8 test=True headless=False checkpoint=runs/smg_gaiting/nn/smg_gaiting.pth
"""

import torch
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import quat_rotate_inverse
from isaacgym.torch_utils import quat_from_angle_axis
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import torch_random_dir
from smg_gym.utils.torch_jit_utils import torch_random_cardinal_dir
from smg_gym.tasks.rewards import compute_reward
from smg_gym.tasks.base_hand import BaseShadowModularGrasper


class SMGGaiting(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):
        """
        Obs:
            joint_pos: 9
            joint_vel: 9
            joint_eff: 9
            fingertip_pos: 9
            fingertip_orn: 12
            latest_action: 9
            prev_action: 9
            bool_tip_contacts: 3
            tip_contact_forces: 9
            ft_sensor_contact_forces: 9
            ft_sensor_contact_torques: 9
            tip_contact_positions: 9
            tip_contact_normals: 9
            tip_contact_force_magnitudes: 3
            object_pos: 3
            object_orn: 4
            object_kps: 18
            object_linvel: 3
            object_angvel: 3
            goal_pos: 3
            goal_orn: 4
            goal_kps: 18
            active_quat: 4
            pivot_axel_vec: 3
            pivot_axel_pos: 3
        max_total = 156
        """

        cfg["env"]["numObservations"] = self.calculate_buffer_size(cfg["enabled_obs"])

        if cfg["asymmetric_obs"]:
            cfg["env"]["numStates"] = self.calculate_buffer_size(cfg["enabled_states"])

        cfg["env"]["numActions"] = self._dims.ActionDim.value

        # target vars
        self.rotate_increment_degrees = cfg["env"]["rotateIncrementDegrees"]
        default_pivot_axel = np.array(cfg["env"]["default_pivot_axel"])
        self.default_pivot_axel = default_pivot_axel/np.linalg.norm(default_pivot_axel)

        # termination vars
        self.max_episode_length = cfg["env"]["episode_length"]

        # task specific randomisation params
        self.rand_pivot_pos = cfg["rand_params"]["rand_pivot_pos"]
        self.rand_pivot_axel = cfg["rand_params"]["rand_pivot_axel"]

        super(SMGGaiting, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

        self._setup_pivot_point()

    def _setup_pivot_point(self):

        # fixed vars
        self.pivot_line_scale = 0.2
        self.pivot_axel_p1 = to_torch(self.default_goal_pos, dtype=torch.float,
                                      device=self.device).repeat((self.num_envs, 1))

        # vars that change on reset (with randomisation)
        self.pivot_point_pos_offset = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_point_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        self.pivot_axel_worldframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_workframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_objframe = torch.zeros(size=(self.num_envs, 3), device=self.device)

        # get indices of pivot points
        self.pivot_point_body_idxs = self._get_pivot_point_idxs(self.envs[0], self.hand_actor_handles[0])

        # amount by which to rotate
        self.rotate_increment = torch.ones(size=(self.num_envs, ), device=self.device) * \
            self.rotate_increment_degrees * np.pi / 180

    def _get_pivot_point_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        body_names = [name for name in hand_body_names if 'pivot_point' in name]
        body_idxs = [self.gym.find_actor_rigid_body_index(
            env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in body_names]

        return body_idxs

    def reset_target_axis(self, env_ids_for_reset):
        """Set target axis to rotate the object about."""

        num_envs_to_reset = len(env_ids_for_reset)

        # get base pose of pivot point.
        pivot_states = self.rigid_body_tensor[env_ids_for_reset, self.pivot_point_body_idxs, :]
        pivot_point_pos = pivot_states[..., 0:3]
        pivot_point_orn = pivot_states[..., 3:7]

        # randomise position of pivot point
        if self.randomize and self.rand_pivot_pos:
            self.pivot_point_pos_offset[env_ids_for_reset, :] = torch_rand_float(
                -0.025, 0.025,
                (num_envs_to_reset, 3),
                device=self.device
            )
        else:
            self.pivot_point_pos_offset[env_ids_for_reset, :] = to_torch(
                [0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))

        self.pivot_point_pos[env_ids_for_reset, :] = pivot_point_pos + self.pivot_point_pos_offset[env_ids_for_reset, :]

        # set the default axel direction
        self.pivot_axel_workframe[env_ids_for_reset, :] = to_torch(
            self.default_pivot_axel,
            dtype=torch.float,
            device=self.device
        ).repeat((num_envs_to_reset, 1))

        # randomise direction of pivot axel
        if self.randomize:
            if self.rand_pivot_axel == 'full_rand':
                self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_dir(
                    num_envs_to_reset,
                    device=self.device
                )
            elif self.rand_pivot_axel == 'cardinal':
                self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_cardinal_dir(
                    num_envs_to_reset,
                    device=self.device
                )

        self.pivot_axel_worldframe[env_ids_for_reset, :] = quat_rotate(
            pivot_point_orn, self.pivot_axel_workframe[env_ids_for_reset, :])

        # find the same pivot axel in the object frame
        obj_base_orn = self.root_state_tensor[self.obj_indices, 3:7]
        self.pivot_axel_objframe[env_ids_for_reset] = quat_rotate_inverse(
            obj_base_orn[env_ids_for_reset], self.pivot_axel_worldframe[env_ids_for_reset, :])

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose to initial pose of the object.
        """

        self.root_state_tensor[
            self.goal_indices[goal_env_ids_for_reset], 3:7
        ] = self.root_state_tensor[self.obj_indices[goal_env_ids_for_reset], 3:7]

        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.root_state_tensor[self.goal_indices, 3:7]

    def rotate_target_pose(self, goal_env_ids_for_reset):
        """
        Rotate the target pose around the pivot axel.
        """
        # rotate goal pose
        rotate_quat = quat_from_angle_axis(self.rotate_increment, self.pivot_axel_objframe)

        self.root_state_tensor[
            self.goal_indices[goal_env_ids_for_reset], 3:7
        ] = quat_mul(self.goal_base_orn, rotate_quat)[goal_env_ids_for_reset, :]
        self.reset_goal_buf[goal_env_ids_for_reset] = 0

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

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_object(env_ids_for_reset)
            self.reset_hand(env_ids_for_reset)
            self.reset_target_pose(env_ids_for_reset)
            self.reset_target_axis(env_ids_for_reset)
            self.rotate_target_pose(env_ids_for_reset)
            actor_root_state_reset_indices.append(self.obj_indices[env_ids_for_reset])
            actor_root_state_reset_indices.append(self.goal_indices[env_ids_for_reset])

        # reset goals
        if len(goal_env_ids_for_reset) > 0:
            self.rotate_target_pose(goal_env_ids_for_reset)
            actor_root_state_reset_indices.append(self.goal_indices[goal_env_ids_for_reset])

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
        self.reset_buf[env_ids_for_reset] = 0
        self.successes[env_ids_for_reset] = 0

    def fill_observation_buffer(self):
        """
        Fill observation buffer.
        shape = (num_envs, num_obs)
        """

        # fill obs buffer with observations shared across tasks.
        obs_cfg = self.cfg["enabled_obs"]
        start_offset, end_offset = self.standard_fill_buffer(self.obs_buf, obs_cfg)

        # target pivot axel vec
        if obs_cfg["pivot_axel_vec"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.VecDim.value
            self.obs_buf[:, start_offset:end_offset] = self.pivot_axel_workframe

        # target pivot axel pos
        if obs_cfg["pivot_axel_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.PosDim.value
            self.obs_buf[:, start_offset:end_offset] = self.pivot_point_pos_offset

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

        # target pivot axel vec
        if states_cfg["pivot_axel_vec"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.VecDim.value
            self.states_buf[:, start_offset:end_offset] = self.pivot_axel_workframe

        # target pivot axel pos
        if states_cfg["pivot_axel_pos"]:
            start_offset = end_offset
            end_offset = start_offset + self._dims.PosDim.value
            self.states_buf[:, start_offset:end_offset] = self.pivot_point_pos_offset

        return self.states_buf

    def compute_reward_and_termination(self):
        """
        Calculate the reward and termination (including goal successes) per env.
        """

        centered_obj_pos = self.obj_base_pos - self.obj_displacement_tensor
        centered_goal_pos = self.goal_base_pos - self.goal_displacement_tensor

        centered_obj_kp_pos = self.obj_kp_positions - \
            self.obj_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1)
        centered_goal_kp_pos = self.goal_kp_positions - \
            self.goal_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1) - \
            self.pivot_point_pos_offset.unsqueeze(1).repeat(1, self.n_keypoints, 1)

        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
            log_dict
        ) = compute_reward(
            # standard
            rew_buf=self.rew_buf,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            reset_goal_buf=self.reset_goal_buf,
            successes=self.successes,
            consecutive_successes=self.consecutive_successes,

            # termination and success criteria
            max_episode_length=self.cfg["env"]["episode_length"],
            fall_reset_dist=self.cfg["env"]["fall_reset_dist"],
            success_tolerance=self.cfg["env"]["success_tolerance"],
            av_factor=self.cfg["env"]["av_factor"],

            # success
            obj_kps=centered_obj_kp_pos,
            goal_kps=centered_goal_kp_pos,
            reach_goal_bonus=self.cfg["env"]["reach_goal_bonus"],
            drop_obj_penalty=self.cfg["env"]["drop_obj_penalty"],

            # precision grasping rew
            n_tip_contacts=self.n_tip_contacts,
            n_non_tip_contacts=self.n_non_tip_contacts,
            require_contact=self.cfg["env"]["require_contact"],
            lamda_good_contact=self.cfg["env"]["lamda_good_contact"],
            lamda_bad_contact=self.cfg["env"]["lamda_bad_contact"],

            # hand smoothness rewards
            actions=self.action_buf,
            current_joint_pos=self.hand_joint_pos,
            current_joint_vel=self.hand_joint_vel,
            current_joint_eff=self.dof_force_tensor,
            init_joint_pos=self._robot_limits["joint_pos"].default,
            lambda_pose_penalty=self.cfg["env"]["lambda_pose_penalty"],
            lambda_torque_penalty=self.cfg["env"]["lambda_torque_penalty"],
            lambda_work_penalty=self.cfg["env"]["lambda_work_penalty"],
            lambda_linvel_penalty=self.cfg["env"]["lambda_linvel_penalty"],

            # obj smoothness
            obj_base_pos=centered_obj_pos,
            goal_base_pos=centered_goal_pos,
            obj_linvel=self.obj_base_linvel,
            current_pivot_axel=quat_rotate(self.obj_base_orn, self.pivot_axel_objframe),
            lambda_com_dist=self.cfg["env"]["lambda_com_dist"],
            lambda_axis_cos_dist=self.cfg["env"]["lambda_axis_cos_dist"],

            # rot reward
            obj_base_orn=self.obj_base_orn,
            goal_base_orn=self.goal_base_orn,
            lambda_rot=self.cfg["env"]["lambda_rot"],
            rot_eps=self.cfg["env"]["rot_eps"],

            # kp reward
            lambda_kp=self.cfg["env"]["lambda_kp"],
            kp_lgsk_scale=self.cfg["env"]["kp_lgsk_scale"],
            kp_lgsk_eps=self.cfg["env"]["kp_lgsk_eps"],

            # angvel reward
            obj_angvel=self.obj_base_angvel,
            target_pivot_axel=self.pivot_axel_worldframe,
            lambda_av=self.cfg["env"]["lambda_av"],
            av_clip_min=self.cfg["env"]["av_clip_min"],
            av_clip_max=self.cfg["env"]["av_clip_max"],
        )

        self.extras.update({"metrics/"+k: v.mean() for k, v in log_dict.items()})

    def visualise_features(self):

        super().visualise_features()

        for i in range(self.num_envs):

            # visualise pivot axel
            pivot_p1 = gymapi.Vec3(
                self.pivot_axel_p1[i, 0],
                self.pivot_axel_p1[i, 1],
                self.pivot_axel_p1[i, 2]
            )

            pivot_axel_p2_worldframe = self.pivot_axel_p1 + self.pivot_axel_worldframe * self.pivot_line_scale
            pivot_p2 = gymapi.Vec3(
                pivot_axel_p2_worldframe[i, 0],
                pivot_axel_p2_worldframe[i, 1],
                pivot_axel_p2_worldframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2,
                gymapi.Vec3(1.0, 1.0, 0.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )

            # visualise object frame pivot_axel
            current_obj_pivot_axel_worldframe = quat_rotate(self.obj_base_orn, self.pivot_axel_objframe)
            pivot_axel_p2_objframe = self.pivot_axel_p1 + current_obj_pivot_axel_worldframe * self.pivot_line_scale

            pivot_p2_objframe = gymapi.Vec3(
                pivot_axel_p2_objframe[i, 0],
                pivot_axel_p2_objframe[i, 1],
                pivot_axel_p2_objframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2_objframe,
                gymapi.Vec3(0.0, 1.0, 1.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )
