import enum
from types import SimpleNamespace
import numpy as np


class AllegroObjectTaskDimensions(enum.Enum):

    # cartesian position + quaternion orientation
    PosDim = 3
    OrnDim = 4
    PoseDim = PosDim + OrnDim

    # linear velocity + angular velcoity
    LinearVelocityDim = 3
    AngularVelocityDim = 3
    VelocityDim = LinearVelocityDim + AngularVelocityDim

    # state: pose + velocity
    StateDim = 13

    # force + torque
    ForceDim = 3
    TorqueDim = 3
    WrenchDim = ForceDim + TorqueDim

    # number of fingers
    NumFingers = 4
    FingertipPosDim = PosDim * NumFingers
    FingertipOrnDim = OrnDim * NumFingers
    FingerContactForceDim = ForceDim * NumFingers
    FingerContactTorqueDim = TorqueDim * NumFingers

    # for three fingers
    NumJointsPerFinger = 4
    ActionDim = NumJointsPerFinger * NumFingers
    JointPositionDim = NumJointsPerFinger * NumFingers
    JointVelocityDim = NumJointsPerFinger * NumFingers
    JointTorqueDim = NumJointsPerFinger * NumFingers

    # for object
    NumKeypoints = 6
    KeypointPosDim = PosDim * NumKeypoints

    # for target
    VecDim = 3


dims = AllegroObjectTaskDimensions

object_properties = {
    "sphere": {
        "radius": 0.035,
        "radius_llim": 0.03,
        "radius_ulim": 0.04,
    },
    "icosphere": {
        "radius": 0.035,
        "radius_llim": 0.03,
        "radius_ulim": 0.04,
    },
    "cube": {
        "size": [0.055, 0.055, 0.055],
        "size_llims": [0.055, 0.055, 0.055],
        "size_ulims": [0.065, 0.065, 0.065],
    },
    "box": {
        "size": [0.055, 0.055, 0.055],
        "size_llims": [0.055, 0.055, 0.055],
        "size_ulims": [0.065, 0.065, 0.065],
    },
    "capsule": {
        "radius": 0.035,
        "radius_llim": 0.03,
        "radius_ulim": 0.04,
        "width": 0.007,
        "width_llim": 0.025,
        "width_ulim": 0.035,
    },
}

robot_dof_properties = {
    "max_position_delta_rad": np.deg2rad(15.0),
    "max_velocity_radps": np.deg2rad(45.0),
    "max_torque_Nm": 0.5,
    "friction": [0.0, 0.0, 0.0, 0.0] * dims.NumFingers.value,
}

# The kp and kd gains of the PD control of the fingers.
robot_dof_gains = {
    "p_gains": [3.0] * dims.ActionDim.value,
    "d_gains": [0.1] * dims.ActionDim.value,
}

# actuated joints on the hand
# For allegro hand hora
control_joint_names = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",         # Index 
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",     # Thumb
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",         # Middle 
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",       # Pinky
]


# limits of the robot (mapped later: str -> torch.tensor)
robot_limits = {
    "joint_pos": SimpleNamespace(
        # matches those on the real robot
        low=np.array([-0.47, -0.196, -0.174, -0.227, 
                      0.263, -0.105, -0.189, -0.162,
                      -0.47, -0.196, -0.174, -0.227, 
                      -0.47, -0.196, -0.174, -0.227], dtype=np.float32),
        high=np.array([0.47, 1.61, 1.709, 1.618,
                      1.396, 1.163, 1.644, 1.719,
                      0.47, 1.61, 1.709, 1.618,
                      0.47, 1.61, 1.709, 1.618], dtype=np.float32),
        default=np.array([0.0, 0.0, 0.0, 0.0] * dims.NumFingers.value, dtype=np.float32),
        # rand_lolim=np.array([-0.2, -0.0, -0.0, -0.0] * (dims.NumFingers.value - 1) + [0.35, 0.55, -0.0, -0.0], dtype=np.float32),
        # rand_uplim=np.array([0.2, 0.465, 0.788, 0.921] * (dims.NumFingers.value - 1) + [0.60, 1.1, 0.788, 0.921] , dtype=np.float32),
        rand_lolim=np.array([-0.3, 0.5, 0.3, -0.227,
                             0.7, 0.4, 0.5, -0.162,
                             -0.2, 0.5, 0.3, -0.227,
                             -0.1, 0.5, 0.3, -0.227], dtype=np.float32),
        rand_uplim=np.array([0.1, 1.5, 1.2, 0.75, 
                             1.396, 1.163, 1.5, 0.7,
                             0.2, 1.5, 1.2, 0.75,
                             0.3, 1.5, 1.2, 0.75] , dtype=np.float32),
    ),
    "joint_vel": SimpleNamespace(
        low=np.full(dims.JointVelocityDim.value, -robot_dof_properties["max_velocity_radps"], dtype=np.float32),
        high=np.full(dims.JointVelocityDim.value, robot_dof_properties["max_velocity_radps"], dtype=np.float32),
        default=np.zeros(dims.JointVelocityDim.value, dtype=np.float32),
    ),
    "joint_eff": SimpleNamespace(
        low=np.full(dims.JointTorqueDim.value, -robot_dof_properties["max_torque_Nm"], dtype=np.float32),
        high=np.full(dims.JointTorqueDim.value, robot_dof_properties["max_torque_Nm"], dtype=np.float32),
        default=np.zeros(dims.JointTorqueDim.value, dtype=np.float32),
    ),
    "fingertip_pos": SimpleNamespace(
        low=np.array([-0.25, -0.25, -0.25] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.25, 0.25, 0.25] * dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_orn": SimpleNamespace(
        low=-np.ones(4 * dims.NumFingers.value, dtype=np.float32),
        high=np.ones(4 * dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_vel": SimpleNamespace(
        low=np.full(dims.VelocityDim.value, -0.25, dtype=np.float32),
        high=np.full(dims.VelocityDim.value, 0.25, dtype=np.float32),
    ),
    "latest_action": SimpleNamespace(
        low=np.full(dims.JointPositionDim.value, -robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
        high=np.full(dims.JointPositionDim.value, robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
    ),
    "prev_action": SimpleNamespace(
        low=np.full(dims.JointPositionDim.value, -robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
        high=np.full(dims.JointPositionDim.value, robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
    ),
    "target_joint_pos": SimpleNamespace(
        low=np.array([-0.47, -0.196, -0.174, -0.227] * (dims.NumFingers.value -1) + [0.263, -0.105, -0.189, -0.162], dtype=np.float32),
        high=np.array([0.47, 1.61, 1.709, 1.618] * (dims.NumFingers.value - 1) + [1.396, 1.163, 1.644, 1.719], dtype=np.float32),
    ),
    "bool_tip_contacts": SimpleNamespace(
        low=np.zeros(dims.NumFingers.value, dtype=np.float32),
        high=np.ones(dims.NumFingers.value, dtype=np.float32),
    ),
    "net_tip_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "ft_sensor_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "ft_sensor_contact_torques": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_positions": SimpleNamespace(
        low=np.array([-0.02, -0.02, -0.02] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.02, 0.02, 0.02] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_normals": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_force_mags": SimpleNamespace(
        low=np.array([0.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([5.0] * dims.NumFingers.value, dtype=np.float32),
    ),
}

object_limits = {
    "object_pos": SimpleNamespace(
        low=np.array([-0.2, -0.2, 0.0], dtype=np.float32),
        high=np.array([0.2, 0.2, 0.4], dtype=np.float32),
    ),
    "object_orn": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "object_kps": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0.0] * dims.NumKeypoints.value, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.5] * dims.NumKeypoints.value, dtype=np.float32),
    ),
    "object_linvel": SimpleNamespace(
        low=np.array([-0.5, -0.5, -2.0], dtype=np.float32),
        high=np.array([0.5, 0.5, 0.2], dtype=np.float32),
    ),
    "object_angvel": SimpleNamespace(
        low=np.full(dims.AngularVelocityDim.value, -1.0, dtype=np.float32),
        high=np.full(dims.AngularVelocityDim.value, 1.0, dtype=np.float32),
    ),
}

goal_limits = {
    "goal_pos": SimpleNamespace(
        low=np.array([-0.2, -0.2, 0.0], dtype=np.float32),
        high=np.array([0.2, 0.2, 0.4], dtype=np.float32),
    ),
    "goal_orn": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "goal_kps": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0.0] * dims.NumKeypoints.value, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.5] * dims.NumKeypoints.value, dtype=np.float32),
    ),
    "active_quat": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "pivot_axel_vec": SimpleNamespace(
        low=np.full(dims.VecDim.value, -1.0, dtype=np.float32),
        high=np.full(dims.VecDim.value, 1.0, dtype=np.float32),
    ),
    "pivot_axel_pos": SimpleNamespace(
        low=np.full(dims.PosDim.value, -0.05, dtype=np.float32),
        high=np.full(dims.PosDim.value, 0.05, dtype=np.float32),
    ),
}
