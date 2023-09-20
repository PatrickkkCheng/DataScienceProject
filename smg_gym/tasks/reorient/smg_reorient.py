"""
Train:
python train.py task=smg_reorient headless=True

# train low env num
python train.py task=smg_reorient task.env.num_envs=8 headless=False train.params.config.horizon_length=1024  train.params.config.minibatch_size=32

Test:
python train.py task=smg_reorient task.env.num_envs=8 test=True headless=False checkpoint=runs/smg_reorient/nn/smg_reorient.pth
"""
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.rewards import compute_reward
from smg_gym.tasks.base_hand import BaseShadowModularGrasper


class SMGReorient(BaseShadowModularGrasper):

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

        super(SMGReorient, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose of the object.
        """
        rand_floats = torch_rand_float(
            -1.0, 1.0,
            (len(goal_env_ids_for_reset), 2),
            device=self.device
        )

        # full rand
        new_goal_quat = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[goal_env_ids_for_reset],
            self.y_unit_tensor[goal_env_ids_for_reset]
        )

        self.root_state_tensor[self.goal_indices[goal_env_ids_for_reset], 3:7] = new_goal_quat
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

        actor_root_state_reset_indices = []

        # nothing to reset
        if len(env_ids_for_reset) == 0 and len(goal_env_ids_for_reset) == 0:
            return None

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_object(env_ids_for_reset)
            self.reset_hand(env_ids_for_reset)
            self.reset_target_pose(env_ids_for_reset)
            actor_root_state_reset_indices.append(self.obj_indices[env_ids_for_reset])
            actor_root_state_reset_indices.append(self.goal_indices[env_ids_for_reset])

        # reset goals
        if len(goal_env_ids_for_reset) > 0:
            self.reset_target_pose(goal_env_ids_for_reset)
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

    def compute_reward_and_termination(self):
        """
        Calculate the reward and termination (including goal successes) per env.
        """

        centered_obj_pos = self.obj_base_pos - self.obj_displacement_tensor
        centered_goal_pos = self.goal_base_pos - self.goal_displacement_tensor

        centered_obj_kp_pos = self.obj_kp_positions - \
            self.obj_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1)
        centered_goal_kp_pos = self.goal_kp_positions - \
            self.goal_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1)

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
            current_pivot_axel=torch.zeros(size=(self.num_envs, 3), device=self.device),
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
            target_pivot_axel=torch.zeros(size=(self.num_envs, 3), device=self.device),
            lambda_av=self.cfg["env"]["lambda_av"],
            av_clip_min=self.cfg["env"]["av_clip_min"],
            av_clip_max=self.cfg["env"]["av_clip_max"],
        )

        self.extras.update({"metrics/"+k: v.mean() for k, v in log_dict.items()})
