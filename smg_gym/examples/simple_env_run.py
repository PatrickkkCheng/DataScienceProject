import gym
import isaacgym
import smg_gym
import torch
import cv2

n_envs = 100
action_type = "static"
envs = smg_gym.make(
	task="allegro_grasp", 
	num_envs=n_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
max_action = torch.from_numpy(envs.action_space.high).to("cuda:0")
min_action = torch.from_numpy(envs.action_space.low).to("cuda:0")
obs = envs.reset()
for _ in range(1000):
	if action_type == "random":
		action = (max_action - min_action) *torch.rand((n_envs,)+envs.action_space.shape, device="cuda:0") + min_action # Random actions
	elif action_type == "static":
		action = torch.zeros((n_envs,)+envs.action_space.shape, device="cuda:0") # No actions#
	else:
		msg = "Invalid action type {}.".format(action_type)
		raise ValueError(msg)
	obs, reward, done, info = envs.step(action)

