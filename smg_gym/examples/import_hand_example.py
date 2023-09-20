import os
import numpy as np
import random
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym.torch_utils import *
import inspect

from smg_gym.assets import get_assets_path, add_assets_path

from smg_gym.tasks.allegro_object_task_params import object_properties
from pybullet_object_models import primitive_objects as object_set

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Import Objects Example")

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.always_use_articulations = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    args.physics_engine,
    sim_params
)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "esc")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "toggle_viewer_sync")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0
plane_params.restitution = 0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 8
grid_size = int(np.sqrt(num_envs))
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create ball asset with gravity disabled from pybullet-object_models
sphere_radius = object_properties["sphere"]['radius']
icosphere_radius = object_properties["sphere"]['radius']
box_size = object_properties['box']['size']
cube_size = object_properties['cube']['size']
capsule_radius = object_properties['capsule']['radius']
capsule_width = object_properties['capsule']['width']

def load_hand():

    # asset_root = add_assets_path('robot_assets/smg_minitip')
    # asset_file = "smg_tactip.urdf"
    # tactip_name = "tactip_tip"

    asset_root = add_assets_path('robot_assets/allegro_hora')
    asset_file = "allegro_digitac.urdf"
    tactip_name = "digitac_tip"

    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.collapse_fixed_joints = False
    asset_options.armature = 0.00001
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.convex_decomposition_from_submeshes = False
    asset_options.vhacd_enabled = False
    asset_options.flip_visual_attachments = False
    asset_options.use_physx_armature = True

    hand_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # set default hand link properties
    hand_props = gym.get_asset_rigid_shape_properties(hand_asset)
    unique_filter = 5
    for p in hand_props:
        p.friction = 2000.0
        p.torsion_friction = 1000.0
        p.rolling_friction = 0.0
        p.restitution = 0.0
        p.filter = unique_filter
        unique_filter += 1
        # p.thickness = 0.001
    gym.set_asset_rigid_shape_properties(hand_asset, hand_props)

    return hand_asset, tactip_name


def load_objects():

    model_list = object_set.getModelList()
    asset_root = object_set.getDataPath()

    object_assets = []
    object_names = []
    for object_name in model_list:
        asset_file = os.path.join(object_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        obj_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        object_assets.append(obj_asset)
        object_names.append(object_name)


    # Create standard objects and add to object assets
    obj_asset = gym.create_sphere(
        sim,
        sphere_radius,
        asset_options
    )
    object_assets.append(obj_asset)
    object_names.append('gym_sphere')

    obj_asset = gym.create_sphere(
        sim,
        icosphere_radius,
        asset_options
    )
    object_assets.append(obj_asset)
    object_names.append('gym_icosphere')

    obj_asset = gym.create_box(
        sim,
        box_size[0],
        box_size[1],
        box_size[2],
        asset_options
    )
    object_assets.append(obj_asset)
    object_names.append('gym_box')

    obj_asset = gym.create_box(
        sim,
        cube_size[0],
        cube_size[1],
        cube_size[2],
        asset_options
    )
    object_assets.append(obj_asset)
    object_names.append('gym_cube')

    obj_asset = gym.create_capsule(
        sim,
        capsule_radius,
        capsule_width,
        asset_options
    )
    object_assets.append(obj_asset)
    object_names.append('gym_capsules')

    return object_assets, object_names


def get_object_asset(asset_object_name):

    return object_assets[object_names.index(asset_object_name)]

# control_joint_names = [
#     "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
#     "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
#     "SMG_F3J1", "SMG_F3J2", "SMG_F3J3",
#     "SMG_F4J1", "SMG_F4J2", "SMG_F4J3",
# ]

# For allegro hand hora
control_joint_names = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",         # Index 
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",         # Middle 
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",       # Pinky
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",     # Thumb
]

num_control_dofs = len(control_joint_names)

mins = {
    "0": -20.0*(np.pi/180),
    "1": -20.0*(np.pi/180),
    "2": -20.0*(np.pi/180),
    "3": -20.0*(np.pi/180),
}
maxs = {
    "0": 20.0*(np.pi/180),
    "1": 20.0*(np.pi/180),
    "2": 20.0*(np.pi/180),
    "3": 20.0*(np.pi/180),
}

def init_hand_joints(env, actor_handle):

    init_joint_pos = {}
    for control_joint in control_joint_names:

        # get rand state for init joint pos
        joint_num = str(int(float(control_joint[6:]) % 4))

        rand_pos = np.random.uniform(mins[joint_num], maxs[joint_num])
        init_joint_pos[control_joint] = rand_pos

        # set this rand pos as the target
        dof_handle = gym.find_actor_dof_handle(env, actor_handle, control_joint)
        gym.set_dof_target_position(env, dof_handle, rand_pos)
        gym.set_dof_target_velocity(env, dof_handle, 0.0)

    # hard reset to random position
    gym.set_actor_dof_states(env, actor_handle, list(init_joint_pos.values()), gymapi.STATE_POS)
    gym.set_actor_dof_states(env, actor_handle, [0.0]*num_control_dofs, gymapi.STATE_VEL)

    return init_joint_pos


def add_hand_actor(env):

    pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
    # pose.r = gymapi.Quat(0, 0, 0, 1)

    up_axis_idx = 2
    pose.p = gymapi.Vec3(*get_axis_params(0.25, up_axis_idx))
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.0*np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)


    num_hand_bodies = gym.get_asset_rigid_body_count(hand_asset)
    num_hand_shapes = gym.get_asset_rigid_shape_count(hand_asset)
    n_hand_dofs = gym.get_asset_dof_count(hand_asset)

    # print("num of hand bodies, ", num_hand_bodies)
    # print("num of hand shapes, ", num_hand_shapes)
    # print("num of hand dofs, ", n_hand_dofs)

    gym.begin_aggregate(env, num_hand_bodies, num_hand_shapes, False)

    handle = gym.create_actor(env, hand_asset, pose, "hand_actor_{}".format(i), 0, -1)

    rigid_body_names = gym.get_actor_rigid_body_names(env, handle)

    # Configure DOF properties
    props = gym.get_actor_dof_properties(env, handle)
    props["driveMode"] = [gymapi.DOF_MODE_POS]*n_hand_dofs
    props["stiffness"] = [5000.0]*n_hand_dofs
    props["damping"] = [100.0]*n_hand_dofs
    gym.set_actor_dof_properties(env, handle, props)

    # Configure actor rigid shape properties
    indices = gym.get_actor_rigid_body_shape_indices(env, handle)
    shape_props = gym.get_actor_rigid_shape_properties(env, handle)

    # new_filter = 5
    # for shape_prop in shape_props:
    #     shape_prop.filter = new_filter
    #     new_filter += 1
    # success = gym.set_actor_rigid_shape_properties(env, handle, shape_props)
    # print('success ', success)
    
    count = 0
    for index in range(len(rigid_body_names)):
        if indices[index].count:
            filter = shape_props[count].filter
            count += 1
        else:
            filter = None
        print("{} {}: count {} start{} collision filter {} ".format(index, rigid_body_names[index], indices[index].count, indices[index].start, filter))

    # create actor handles
    control_handles = {}
    for control_joint in control_joint_names:
        dof_handle = gym.find_actor_dof_handle(env, handle, control_joint)
        control_handles[control_joint] = dof_handle

    init_joint_pos = init_hand_joints(env, handle)

    gym.end_aggregate(env)

    return handle, control_handles, init_joint_pos


def add_object_actor(env):
    pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(0.0, 0.0, 0.275)
    # pose.r = gymapi.Quat(0, 0, 0, 1)

    pose.p = gymapi.Vec3(0.0, -0.02, 0.36)
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.0*np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * np.pi) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.5 * np.pi)

    # object_asset = np.random.choice(object_assets)
    object_asset = get_object_asset('gym_sphere')
    handle = gym.create_actor(env, object_asset, pose, "obj_actor_{}".format(i), 0, 2)

    obj_props = gym.get_actor_rigid_body_properties(env, handle)
    obj_props[0].mass = 1.0
    gym.set_actor_rigid_body_properties(env, handle, obj_props)

    # set color of object
    env_obj_color = gymapi.Vec3(1, 0, 0)
    gym.set_rigid_body_color(
        env,
        handle,
        0,
        gymapi.MESH_VISUAL,
        env_obj_color
    )

    # set shape properties (some settings only work on FLEX backend)
    shape_props = gym.get_actor_rigid_shape_properties(env, handle)
    gym.set_actor_rigid_shape_properties(env, handle, shape_props)

    obj_body_name = gym.get_actor_rigid_body_names(env, handle)
    obj_body_idx = gym.find_actor_rigid_body_index(env, handle, obj_body_name[0], gymapi.DOMAIN_ENV)

    return handle, object_asset


def get_object_state(env, obj_actor_handle):

    obj_state = gym.get_actor_rigid_body_states(env, obj_actor_handle, gymapi.STATE_ALL)
    pos = obj_state['pose']['p']
    orn = obj_state['pose']['r']
    lin_vel = obj_state['vel']['linear']
    ang_vel = obj_state['vel']['angular']

    return pos, orn, lin_vel, ang_vel

def apply_gravity_compensation_object(env, obj_actor_handle):

    obj_props = gym.get_actor_rigid_body_properties(env, obj_actor_handle)
    mass = obj_props[0].mass
    gravity = sim_params.gravity
    # gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    force = -(gravity * mass)

    body_names = gym.get_actor_rigid_body_names(env, obj_actor_handle)
    rigid_body_handle = gym.find_actor_rigid_body_handle(env, obj_actor_handle, body_names[0])
    gym.apply_body_forces(env, rigid_body_handle, force=force, torque=None, space=gymapi.ENV_SPACE)


def pre_physics_step():

    act_lim = 0.1
    dof_speed_scale = 10.0
    dt = sim_params.dt

    for i in range(num_envs):

        # action = np.random.uniform(-act_lim, act_lim, size=[num_control_dofs])
        action = np.zeros(num_control_dofs)

        current_joint_state = current_joint_states[i]
        targets = current_joint_state + dof_speed_scale * dt * action

        for (j, dof_handle) in enumerate(hand_control_joint_handles[i].values()):
            gym.set_dof_target_position(envs[i], dof_handle, targets[j])

        current_joint_states[i] = targets


def post_physics_step():

    gym.refresh_net_contact_force_tensor(sim)

    pos, orn, lin_vel, ang_vel = get_object_state(envs[i], object_actor_handles[i])

    # Printing out contacts
    # contacts = get_tip_contacts(envs[0], hand_actor_handles[0], object_actor_handles[0])
    # print(contacts)

    net_tip_contact_forces, tip_object_contacts, n_tip_contacts, n_non_tip_contacts = get_fingertip_contacts()

    if any(tip_object_contacts[0]):
        print("tip_contacted with n: ", tip_object_contacts[0])

    if n_non_tip_contacts[0] > 0:
        print("Non tip_contacted with : ", n_non_tip_contacts[0])


def apply_grasp_action(current_joint_states):

    act_lim = 0.1
    dof_speed_scale = 10.0
    dt = sim_params.dt

    grasp_action = np.array([
        0.0, act_lim, act_lim, 0.0,
        0.0, act_lim, act_lim, 0.0,
        0.0, act_lim, act_lim, 0.0,
        act_lim, act_lim, act_lim, 0.0,
    ])

    for i in range(num_envs):
        current_joint_state = current_joint_states[i]
        targets = current_joint_state + dof_speed_scale * dt * grasp_action

        for (j, dof_handle) in enumerate(hand_control_joint_handles[i].values()):
            gym.set_dof_target_position(envs[i], dof_handle, targets[j])

        current_joint_states[i] = targets


def get_tip_contacts(env, hand_actor_handle, obj_actor_handle):

    contacts = gym.get_env_rigid_contacts(env)
    # contact_forces = gym.get_env_rigid_contact_forces(env)

    obj_body_names = gym.get_actor_rigid_body_names(env, obj_actor_handle)
    obj_body_idx = gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_names[0], gymapi.DOMAIN_ENV)

    hand_body_names = gym.get_actor_rigid_body_names(env, hand_actor_handle)
    tip_body_names = [name for name in hand_body_names if tactip_name in name]
    tip_body_idxs = [gym.find_actor_rigid_body_index(
        env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names]

    tip_contacts = [False] * len(tip_body_idxs)

    # iterate through contacts and update tip_contacts list if there is a
    # contact between object and tip
    for contact in contacts:
        if obj_body_idx in [contact['body0'], contact['body1']]:
            current_tip_contacts = [tip_body_idx in [contact['body0'], contact['body1']] for tip_body_idx in tip_body_idxs]

            for i, current_tip_contact in enumerate(current_tip_contacts):
                if current_tip_contact:
                    tip_contacts[i] = True

    return tip_contacts

def get_fingertip_contacts():

    hand_body_names = gym.get_asset_rigid_body_names(hand_asset)
    tip_body_names = [name for name in hand_body_names if tactip_name in name]
    non_tip_body_names = [name for name in hand_body_names if tactip_name not in name]
    tip_body_idxs = [
        gym.find_asset_rigid_body_index(hand_asset, name) for name in tip_body_names
    ]
    non_tip_body_idxs = [
        gym.find_asset_rigid_body_index(hand_asset, name) for name in non_tip_body_names
    ]
    n_tips = len(tip_body_idxs)
    n_non_tip_links = len(non_tip_body_idxs)

    # get envs where obj is contacted
    bool_obj_contacts = torch.where(
        torch.count_nonzero(contact_force_tensor[:, obj_body_idx, :], dim=1) > 0,
        torch.ones(size=(num_envs,)),
        torch.zeros(size=(num_envs,)),
    )

    # get envs where tips are contacted
    net_tip_contact_forces = contact_force_tensor[:, tip_body_idxs, :]
    bool_tip_contacts = torch.where(
        torch.count_nonzero(net_tip_contact_forces, dim=2) > 0,
        torch.ones(size=(num_envs, n_tips)),
        torch.zeros(size=(num_envs, n_tips)),
    )

    # get all the contacted links that are not the tip
    net_non_tip_contact_forces = contact_force_tensor[:, non_tip_body_idxs, :]
    bool_non_tip_contacts = torch.where(
        torch.count_nonzero(net_non_tip_contact_forces, dim=2) > 0,
        torch.ones(size=(num_envs, n_non_tip_links)),
        torch.zeros(size=(num_envs, n_non_tip_links)),
    )
    n_non_tip_contacts = torch.sum(bool_non_tip_contacts, dim=1)

    # repeat for n_tips shape=(n_envs, n_tips)
    onehot_obj_contacts = bool_obj_contacts.unsqueeze(1).repeat(1, n_tips)

    # get envs where object and tips are contacted
    tip_object_contacts = torch.where(
        onehot_obj_contacts > 0,
        bool_tip_contacts,
        torch.zeros(size=(num_envs, n_tips))
    )
    n_tip_contacts = torch.sum(bool_tip_contacts, dim=1)

    return net_tip_contact_forces, tip_object_contacts, n_tip_contacts, n_non_tip_contacts


def initialise_contact(current_joint_states):
    max_steps = 50
    update_envs = list(range(num_envs))

    for i in range(max_steps):

        if update_envs == []:
            break

        # for i in update_envs:
        #     # Disable gravity during grasping motion
        #     apply_gravity_compensation_object(envs[i], object_actor_handles[i])
        #     # apply_grasp_action(current_joint_states)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        for i in update_envs:
            tip_contacts = get_tip_contacts(envs[i], hand_actor_handles[i], object_actor_handles[i])

            # if all three tips establish contact then stop appying grasp action
            if sum(tip_contacts) == 3:
                update_envs.remove(i)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


def reset():
    gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # set rand init start state
    init_joint_poses = []
    for env, handle in zip(envs, hand_actor_handles):
        init_joint_pos = init_hand_joints(env, handle)
        init_joint_poses.append(init_joint_pos)

    current_joint_states = []
    for i in range(num_envs):
        current_joint_states.append(np.array(list(init_joint_poses[i].values())))

    initialise_contact(current_joint_states)

    return current_joint_states


# create hand asset
hand_asset, tactip_name = load_hand()
object_assets, object_names = load_objects()

# hand_body_names = gym.get_asset_rigid_body_names(hand_asset)
# tcp_body_names = [name for name in hand_body_names if "tcp" in name]

# create list to mantain environment and asset handles
envs = []
hand_actor_handles = []
hand_control_joint_handles = []
init_joint_poses = []
object_actor_handles = []
object_asset_list = []
for i in range(num_envs):

    env = gym.create_env(sim, env_lower, env_upper, grid_size)

    hand_actor_handle, hand_control_handles, init_joint_pos = add_hand_actor(env)
    object_handle, object_asset = add_object_actor(env)

    envs.append(env)
    hand_actor_handles.append(hand_actor_handle)
    hand_control_joint_handles.append(hand_control_handles)
    init_joint_poses.append(init_joint_pos)
    object_actor_handles.append(object_handle)
    object_asset_list.append(object_asset)

    obj_body_name = gym.get_actor_rigid_body_names(env, hand_actor_handle)
    obj_body_idx = gym.find_actor_rigid_body_index(env, object_handle, obj_body_name[0], gymapi.DOMAIN_ENV)


# print('Envs: ', envs)
# print('Actors: ', hand_actor_handles)
# print('Joints: ', hand_control_joint_hanles)
# print('Init Joints: ', init_joint_poses)

# get useful numbers
n_sim_bodies = gym.get_sim_rigid_body_count(sim)
n_env_bodies = gym.get_sim_rigid_body_count(sim) // num_envs

# Set up some tensors
contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)    
contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor).view(num_envs, n_env_bodies, 3)

# look at the first env
cam_pos = gymapi.Vec3(2, 2, 2)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# save initial state for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]

current_joint_states = reset()

# clear lines every n steps
clear_step = 25
step_counter = 0
enable_viewer_sync = True

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        
        if evt.value > 0:
            print(evt.action)

        if evt.action == "reset" and evt.value > 0:
            current_joint_states = reset()

        if ((evt.action == "quit" and evt.value > 0)
                or (evt.action == "esc" and evt.value > 0)):
            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            quit()

        if evt.action == "toggle_viewer_sync" and evt.value > 0:
            enable_viewer_sync = not enable_viewer_sync

    # apply a step before simulating
    pre_physics_step()

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # apply a step after simulating
    post_physics_step()

    # update the viewer
    if enable_viewer_sync:

        # draw contacts for first env
        # gym.draw_env_rigid_contacts(viewer, envs[0], colors[0], 0.25, True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

        # remove all lines drawn on viewer
        if step_counter % clear_step == 0:
            gym.clear_lines(viewer)

    else:
        gym.poll_viewer_events(viewer)

    step_counter += 1

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
