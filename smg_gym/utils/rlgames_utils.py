# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Callable

from smg_gym.tasks import task_map


def get_rlgames_env_creator(
        # used to create the vec task
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env(_sim_device=sim_device, _rl_device=rl_device, **kwargs):
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        if multi_gpu:
            import horovod.torch as hvd

            rank = hvd.rank()
            print("Horovod rank: ", rank)

            _sim_device = f'cuda:{rank}'
            # _rl_device = f'cuda:{rank}'

            task_config['rank'] = rank
            task_config['rl_device'] = 'cuda:' + str(rank)
        else:
            _sim_device = sim_device
            # _rl_device = rl_device

        # create native task and pass custom config
        env = task_map[task_name](
            cfg=task_config,
            sim_device=_sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env
