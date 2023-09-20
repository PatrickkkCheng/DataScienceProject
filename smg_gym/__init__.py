import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import smg_gym

# OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
# OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
# OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def make(
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    cfg: DictConfig = None
): 
    from smg_gym.utils.rlgames_utils import get_rlgames_env_creator
    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices['task']
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./rl_games_helpers/cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['num_envs'] = num_envs
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    create_rlgpu_env = get_rlgames_env_creator(
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
    )
    return create_rlgpu_env()
