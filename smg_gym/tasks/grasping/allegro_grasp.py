"""
Train:
python gen_stable_grasps.py task=smg_grasp headless=True

# train low env num
python gen_stable_grasps.py task=smg_grasp task.env.num_envs=8 headless=False train.params.config.horizon_length=1024  train.params.config.minibatch_size=32

Test:
python gen_stable_grasps.py task=smg_grasp task.env.num_envs=8 test=True headless=False checkpoint=runs/smg_reorient/nn/smg_reorient.pth
"""
import os
import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.rewards import compute_stable_grasp
from smg_gym.tasks.base_allegro import BaseAllegro


class AllegroGrasp(BaseAllegro):

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

        super(AllegroGrasp, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

        self.saved_grasping_states = torch.zeros(
            (0, self._dims.JointPositionDim.value + self._dims.PoseDim.value),
            dtype=torch.float,
            device=self.device
        )

    def save_successful_grasps(self, env_ids_for_reset):

        # find stable grasps that lasted until the episode length
        success = self.progress_buf[env_ids_for_reset] == self.max_episode_length

        # concatenate joint positions and object positions
        all_states = torch.cat([
            self.dof_pos[:, :].squeeze(),
            self.root_state_tensor[self.obj_indices, 0:7],
        ], dim=1)

        # accumulate stable grasps
        self.saved_grasping_states = torch.cat([
            self.saved_grasping_states,
            all_states[env_ids_for_reset][success]
        ])

        # once cache has reached maximum size, save as numpy file
        print('Current cache size: {}'.format(self.saved_grasping_states.shape[0]))
        max_size = 1000
        if len(self.saved_grasping_states) >= max_size:
            print('Saving cache of stable grasps')
            save_file = os.path.join(
                os.path.dirname(__file__),
                '../../saved_grasps/allegro/',
                f'{self.obj_name}_grasps.npy'
            )
            np.save(save_file, self.saved_grasping_states[:max_size].cpu().numpy())
            exit()

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

        # don't save grasps if using cached grasps for reset
        if not self.cfg["env"]["use_cached_grasps"]:
            self.save_successful_grasps(env_ids_for_reset)
        # randomly select from loaded grasps
        else: 
            self.sampled_pose_idx = np.random.randint(self.loaded_grasping_states.shape[0], size=len(env_ids_for_reset))

        actor_root_state_reset_indices = []

        # nothing to reset
        if len(env_ids_for_reset) == 0 and len(goal_env_ids_for_reset) == 0:
            return None

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_object(env_ids_for_reset)
            self.reset_hand(env_ids_for_reset)
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

        # if any(self.tip_object_contacts[0]):
        #     print("Env 0 tip_contacted with n: ", self.tip_object_contacts[0])

        # if self.n_non_tip_contacts[0] > 0:
        #     print(" Env 0 Non tip_contacted with : ", self.n_non_tip_contacts[0])


        # print(" Env 0 is thumb tip in contact : ", self.thumb_tip_contacts[0])

        # print(['%.3f' % elem for elem in self.hand_joint_pos[8:12]])

        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
            log_dict
        ) = compute_stable_grasp(
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

            # stable grasp components
            thumb_tip_contacts=self.thumb_tip_contacts,
            n_tip_contacts=self.n_tip_contacts,
            n_non_tip_contacts=self.n_non_tip_contacts,
            obj_base_pos=centered_obj_pos,
            goal_base_pos=centered_goal_pos,
        )

        # if self.reset_buf[0]:
        #     print("Env 0 resetted at step {}".format(self.progress_buf[0]))

        self.extras.update({"metrics/"+k: v.mean() for k, v in log_dict.items()})
