# import os
# import numpy as np
# import random
# from isaacgym import gymutil
# from isaacgym import gymapi

# from pybullet_object_models import primitive_objects as object_set
# # from pybullet_object_models import random_objects as object_set
# # from pybullet_object_models import ycb_objects as object_set
# # from pybullet_object_models import superquadric_objects as object_set
# # from pybullet_object_models import google_scanned_objects as object_set
# # from pybullet_object_models import gibson_feelies as object_set
# # from pybullet_object_models import gibson_glavens as object_set
# # from pybullet_object_models import gibson_bellpeppers as object_set
# # from pybullet_object_models.egad import egad_train_set as object_set
# # from pybullet_object_models.egad import egad_eval_set as object_set

# # initialize gym
# gym = gymapi.acquire_gym()

# # parse arguments
# args = gymutil.parse_arguments(description="Import Objects Example")

# # configure sim
# sim_params = gymapi.SimParams()
# if args.physics_engine == gymapi.SIM_FLEX:
#     sim_params.flex.relaxation = 0.9
#     sim_params.flex.dynamic_friction = 0.0
#     sim_params.flex.static_friction = 0.0
# elif args.physics_engine == gymapi.SIM_PHYSX:
#     sim_params.physx.solver_type = 1
#     sim_params.physx.num_position_iterations = 4
#     sim_params.physx.num_velocity_iterations = 1
#     sim_params.physx.num_threads = args.num_threads
#     sim_params.physx.use_gpu = args.use_gpu

# # sim_params.use_gpu_pipeline = False
# # if args.use_gpu_pipeline:
# #     print("WARNING: Forcing CPU pipeline.")

# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)


# sim = gym.create_sim(
#     args.compute_device_id,
#     args.graphics_device_id,
#     args.physics_engine,
#     sim_params
# )

# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# # create viewer using the default camera properties
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise ValueError('*** Failed to create viewer')

# gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
# gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")

# # add ground plane
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
# plane_params.distance = 0
# plane_params.static_friction = 0.0
# plane_params.dynamic_friction = 0.0
# plane_params.restitution = 0

# gym.add_ground(sim, plane_params)

# # set up the env grid
# num_envs = 16
# grid_size = 16
# spacing = 0.5
# env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
# env_upper = gymapi.Vec3(spacing, spacing, spacing)

# # create list to mantain environment and asset handles
# envs = []
# actor_handles = []

# # create ball asset with gravity disabled from pybullet-object_models


# def load_object():

#     model_list = object_set.getModelList()
#     object_name = random.choice(model_list)

#     asset_root = object_set.getDataPath()
#     asset_file = os.path.join(object_name, "model.urdf")

#     asset_options = gymapi.AssetOptions()
#     asset_options.disable_gravity = False
#     asset_options.fix_base_link = False

#     asset_obj = gym.load_asset(sim, asset_root, asset_file, asset_options)

#     return asset_obj

# # add object actor


# def add_object_actor(env):

#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
#     pose.r = gymapi.Quat(0, 0, 0, 1)

#     obj_handle = gym.create_actor(env, asset_obj, pose, "actor_obj_{}".format(i), -1, -1)

#     actor_handles.append(obj_handle)


# # create env
# for i in range(num_envs):
#     env = gym.create_env(sim, env_lower, env_upper, grid_size)
#     envs.append(env)
#     asset_obj = load_object()
#     add_object_actor(envs[i])

# # look at the first env
# cam_pos = gymapi.Vec3(6, 4.5, 3)
# cam_target = gymapi.Vec3(-0.8, 0.5, 0)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # save initial state for reset
# initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# while not gym.query_viewer_has_closed(viewer):

#     # Get input actions from the viewer and handle them appropriately
#     for evt in gym.query_viewer_action_events(viewer):

#         if evt.action == "reset" and evt.value > 0:
#             gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

#         if evt.action == "quit" and evt.value > 0:
#             gym.destroy_viewer(viewer)
#             gym.destroy_sim(sim)
#             quit()

#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)

# print('Done')

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)

import os
import numpy as np
import random
from isaacgym import gymutil
from isaacgym import gymapi

from pybullet_object_models import primitive_objects as object_set
# from pybullet_object_models import random_objects as object_set
# from pybullet_object_models import ycb_objects as object_set
# from pybullet_object_models import superquadric_objects as object_set
# from pybullet_object_models import google_scanned_objects as object_set
# from pybullet_object_models import gibson_feelies as object_set
# from pybullet_object_models import gibson_glavens as object_set
# from pybullet_object_models import gibson_bellpeppers as object_set
# from pybullet_object_models.egad import egad_train_set as object_set
# from pybullet_object_models.egad import egad_eval_set as object_set

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
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

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

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0
plane_params.restitution = 0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 16
grid_size = 16
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
actor_handles = []

# create ball asset with gravity disabled from pybullet-object_models


def load_object():

    model_list = object_set.getModelList()
    object_name = random.choice(model_list)

    asset_root = object_set.getDataPath()
    asset_file = os.path.join(object_name, "model.urdf")

    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = False
    asset_options.fix_base_link = False

    asset_obj = gym.load_asset(sim, asset_root, asset_file, asset_options)

    return asset_obj

# add object actor


def add_object_actor(env):

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    obj_handle = gym.create_actor(env, asset_obj, pose, "actor_obj_{}".format(i), -1, -1)

    actor_handles.append(obj_handle)


# create env
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, grid_size)
    envs.append(env)
    asset_obj = load_object()
    add_object_actor(envs[i])

# look at the first env
cam_pos = gymapi.Vec3(6, 4.5, 3)
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# save initial state for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


# Initialize a timer to control the gravity changes
timer = 0
period = 1.0  # Change gravity every 1 second

# Main simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

        if evt.action == "quit" and evt.value > 0:
            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            quit()

    # Increment the timer
    timer += sim_params.dt

    # Change the gravity direction every 'period' seconds
    if timer >= period:
        # Generate random values for the gravity direction
        random_gravity = gymapi.Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        random_gravity.normalize()  # Normalize to get a unit vector
        sim_params.gravity = random_gravity * 9.8  # Apply the desired magnitude
        timer = 0  # Reset the timer

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
