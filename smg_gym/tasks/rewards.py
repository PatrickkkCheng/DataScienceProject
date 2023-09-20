from typing import Tuple, Dict
import torch

from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate

from smg_gym.utils.torch_jit_utils import lgsk_kernel


@torch.jit.script
def compute_reward(
    # standard
    rew_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,

    # termination and success criteria
    max_episode_length: float,
    fall_reset_dist: float,
    success_tolerance: float,
    av_factor: float,

    # success
    obj_kps: torch.Tensor,
    goal_kps: torch.Tensor,
    reach_goal_bonus: float,
    drop_obj_penalty: float,

    # precision grasping rew
    n_tip_contacts: torch.Tensor,
    n_non_tip_contacts: torch.Tensor,
    require_contact: bool,
    lamda_good_contact: float,
    lamda_bad_contact: float,

    # hand smoothness rewards
    actions: torch.Tensor,
    current_joint_pos: torch.Tensor,
    current_joint_vel: torch.Tensor,
    current_joint_eff: torch.Tensor,
    init_joint_pos: torch.Tensor,
    lambda_pose_penalty: float,
    lambda_torque_penalty: float,
    lambda_work_penalty: float,
    lambda_linvel_penalty: float,

    # obj smoothness reward
    obj_base_pos: torch.Tensor,
    goal_base_pos: torch.Tensor,
    obj_linvel: torch.Tensor,
    current_pivot_axel: torch.Tensor,
    lambda_com_dist: float,
    lambda_axis_cos_dist: float,

    # hybrid reward
    obj_base_orn: torch.Tensor,
    goal_base_orn: torch.Tensor,
    lambda_rot: float,
    rot_eps: float,

    # kp reward
    lambda_kp: float,
    kp_lgsk_scale: float,
    kp_lgsk_eps: float,

    # angvel reward
    obj_angvel: torch.Tensor,
    target_pivot_axel: torch.Tensor,
    lambda_av: float,
    av_clip_min: float,
    av_clip_max: float,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # ROTATION REWARD
    # cosine distance between obj and goal orientation
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(goal_base_orn))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    rot_rew = (1.0 / (torch.abs(rot_dist) + rot_eps))

    # KEYPOINT REWARD
    # distance between obj and goal keypoints
    kp_deltas = torch.norm(obj_kps - goal_kps, p=2, dim=-1)
    mean_kp_dist = kp_deltas.mean(dim=-1)
    kp_rew = lgsk_kernel(kp_deltas, scale=kp_lgsk_scale, eps=kp_lgsk_eps).mean(dim=-1)

    # ANGVEL REWARD
    # bound and scale rewards such that they are in similar ranges
    obj_angvel_about_axis = torch.sum(obj_angvel * target_pivot_axel, dim=1)
    av_rew = torch.clamp(obj_angvel_about_axis, min=av_clip_min, max=av_clip_max)

    # HAND SMOOTHNESS
    # Penalty for deviating from the original grasp pose by too much
    hand_pose_penalty = -torch.norm(current_joint_pos - init_joint_pos, p=2, dim=-1)
    # Penalty for high torque
    torque_penalty = -torch.norm(current_joint_eff, p=2, dim=-1)
    # Penalty for high work
    work_penalty = -torch.sum(torch.abs(current_joint_eff * current_joint_vel), dim=-1)

    # OBJECT SMOOTHNESS
    # distance between obj and goal COM
    com_dist_rew = -torch.norm(obj_base_pos - goal_base_pos, p=2, dim=-1)
    # Penalty for object linear velocity
    obj_linvel_penalty = -torch.norm(obj_linvel, p=2, dim=-1)
    # Penalty for axis deviation
    axis_cos_dist = -(1.0 - torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12))

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = \
        lambda_rot * rot_rew + \
        lambda_kp * kp_rew + \
        lambda_av * av_rew + \
        lambda_pose_penalty * hand_pose_penalty + \
        lambda_torque_penalty * torque_penalty + \
        lambda_work_penalty * work_penalty + \
        lambda_com_dist * com_dist_rew + \
        lambda_linvel_penalty * obj_linvel_penalty + \
        lambda_axis_cos_dist * axis_cos_dist

    # PRECISION GRASP
    # add reward for contacting with tips
    total_reward = torch.where(n_tip_contacts >= 2, total_reward + lamda_good_contact, total_reward)

    # add penalty for contacting with links other than the tips
    total_reward = torch.where(n_non_tip_contacts > 0, total_reward - lamda_bad_contact, total_reward)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    total_reward = torch.where(mean_kp_dist <= success_tolerance, total_reward + reach_goal_bonus, total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    # print(successes)
    total_reward = torch.where(mean_kp_dist >= fall_reset_dist, total_reward - drop_obj_penalty, total_reward)

    # zero reward when less than 2 tips in contact
    if require_contact:
        total_reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), total_reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(mean_kp_dist <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(mean_kp_dist >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # find average consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    info: Dict[str, torch.Tensor] = {

        'successes': successes,
        'successes_cons': cons_successes,

        'num_tip_contacts': n_tip_contacts,
        'num_non_tip_contacts': n_non_tip_contacts,

        'reward_rot': rot_rew,
        'reward_keypoint': kp_rew,
        'reward_angvel': av_rew,
        'reward_total': total_reward,

        'penalty_hand_pose': hand_pose_penalty,
        'penalty_hand_torque': torque_penalty,
        'penalty_hand_work': work_penalty,

        'reward_com_dist': com_dist_rew,
        'penalty_obj_linvel': obj_linvel_penalty,
        'penalty_axis_cos_dist': axis_cos_dist,
    }

    return total_reward, resets, goal_resets, successes, cons_successes, info


@torch.jit.script
def compute_stable_grasp(
    # standard
    rew_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,

    # termination and success criteria
    max_episode_length: float,
    fall_reset_dist: float,

    # stable grasp components
    thumb_tip_contacts: torch.Tensor,
    n_tip_contacts: torch.Tensor,
    n_non_tip_contacts: torch.Tensor,
    obj_base_pos: torch.Tensor,
    goal_base_pos: torch.Tensor,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    total_reward = torch.zeros_like(rew_buf)
    resets = torch.zeros_like(reset_buf)
    goal_resets = torch.zeros_like(reset_buf)
    successes = torch.zeros_like(successes)
    cons_successes = torch.zeros_like(consecutive_successes)

    # com dist to detect fall
    com_dist_rew = -torch.norm(obj_base_pos - goal_base_pos, p=2, dim=-1)
    resets = torch.where(com_dist_rew >= fall_reset_dist, torch.ones_like(reset_buf), resets)

    # number of contacts to detect grasp
    resets = torch.where(n_tip_contacts < 2, torch.ones_like(reset_buf), resets)

    # thumb in contact 
    resets = torch.where(thumb_tip_contacts == 0, torch.ones_like(reset_buf), resets)

    # number of non-tip contacts to detect bad grasp
    resets = torch.where(n_non_tip_contacts >= 2, torch.ones_like(reset_buf), resets)

    # max ep length is reached
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'num_tip_contacts': n_tip_contacts,
        'num_non_tip_contacts': n_non_tip_contacts,
    }

    return total_reward, resets, goal_resets, successes, cons_successes, info
