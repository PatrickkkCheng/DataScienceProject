from smg_gym.utils.rlgames_utils import get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from hydra.utils import to_absolute_path

import hydra
import isaacgym
import os
import ray
import yaml
import torch
import gym
import numpy as np
import onnx
import onnxruntime as ort

from rl_games.torch_runner import Runner
import rl_games.algos_torch.flatten as flatten

from omegaconf import OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)

# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

# load config
saved_model_path = 'runs/smg_gaiting/dr_on'

config_name = os.path.join(
    saved_model_path,
    'config.yaml'
)
# load config for real env with pointer to trained agent
cfg = OmegaConf.load(
    config_name
)

# ensure checkpoints can be specified as relative paths
if cfg.checkpoint:
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

cfg_dict = omegaconf_to_dict(cfg)
print_dict(cfg_dict)

# set numpy formatting for printing only
set_np_formatting()

# sets seed. if seed is -1 will pick a random one
cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

# `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
# We use the helper function here to specify the environment config.
create_rlgpu_env = get_rlgames_env_creator(
    omegaconf_to_dict(cfg.task),
    cfg.task_name,
    cfg.sim_device,
    cfg.rl_device,
    cfg.graphics_device_id,
    cfg.headless,
    multi_gpu=cfg.multi_gpu,
)

# register the rl-games adapter to use inside the runner
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
})

rlg_config_dict = omegaconf_to_dict(cfg.train)

# convert CLI arguments into dictionory
# create runner and set the settings
runner = Runner(RLGPUAlgoObserver())
runner.load(rlg_config_dict)

# restort model
agent = runner.create_player()
agent.restore(os.path.join(
    saved_model_path, 'nn', 'smg_gaiting.pth'
))

inputs = {
    'is_train': False,
    'prev_actions': None,
    'obs': torch.zeros((1,) + agent.obs_shape).to(agent.device),
    'rnn_states': agent.states
}

with torch.no_grad():
    adapter = flatten.TracingAdapter(agent.model.a2c_network, inputs, allow_non_tensor=True)
    traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
    flattened_outputs = traced(*adapter.flattened_inputs)
    print(flattened_outputs)

torch.onnx.export(
    traced,
    *adapter.flattened_inputs,
    os.path.join(
        saved_model_path, 'nn', 'smg_gaiting.onnx'
    ),
    verbose=True,
    input_names=['obs'],
    output_names=['logits', 'value']
)
