import random
import numpy as np
import operator
from copy import deepcopy

import torch

from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args
from isaacgym.gymutil import apply_random_samples, check_buckets, generate_random_samples


class DomainRandomizer:
    """
    Methods for applying domain randomization.
    """

    def __init__(self, sim, gym, envs, apply_dr, dr_params, num_envs):

        self.sim = sim
        self.gym = gym
        self.envs = envs
        self.apply_dr = apply_dr
        self.dr_params = dr_params
        self.num_envs = num_envs

        print('')
        print(f'Domain Randomization: {self.apply_dr}')
        for key in self.dr_params:
            print(key, self.dr_params[key])
        print('')

        self.nonphysical_dr_randomizations = {}
        self.first_randomization = True
        self.original_props = {}
        self.actor_params_generator = None
        self.last_step = -1
        self.last_rand_step = -1

        self.extern_actor_params = {}
        for env_id in range(num_envs):
            self.extern_actor_params[env_id] = None

    def get_actor_params_info(self, env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in self.dr_params:
            return None

        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)

        for actor, actor_properties in self.dr_params["actor_params"].items():

            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():

                if prop_name == 'color':
                    continue  # this is set randomly
                if prop_name == 'scale':
                    continue  # not in param_getters_map (may need to adjust get_property_getter_map())

                props = param_getters_map[prop_name](env, handle)

                if not isinstance(props, list):
                    props = [props]

                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():

                        name = prop_name + '_' + str(prop_idx) + '_' + attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']

                        # if 'uniform' not in distr:
                        if distr not in ['uniform', 'gaussian']:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))

                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])

        return params, names, lows, highs

    def apply_observation_randomization(self, observations):
        if self.nonphysical_dr_randomizations.get('observations', None):
            observations = self.nonphysical_dr_randomizations['observations']['noise_lambda'](observations)
        return observations

    def apply_action_randomization(self, actions):
        if self.nonphysical_dr_randomizations.get('actions', None):
            actions = self.nonphysical_dr_randomizations['actions']['noise_lambda'](actions)
        return actions

    def apply_domain_randomization(
        self,
        randomize_buf,
        reset_buf,
        sim_initialized
    ):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            sim_initialized: flag for whether the sim has already been initialized.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = self.dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - on the first call, randomize everything
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(
                randomize_buf >= rand_freq,
                torch.ones_like(randomize_buf),
                torch.zeros_like(randomize_buf)
            )
            rand_envs = torch.logical_and(rand_envs, reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, self.dr_params)

        # create dict with functions for applying randomisations to non-physical parameters
        # (observation / action noise)
        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in self.dr_params and do_nonenv_randomize:

                dist = self.dr_params[nonphysical_param]["distribution"]
                op_type = self.dr_params[nonphysical_param]["operation"]
                sched_type = self.dr_params[nonphysical_param]["schedule"] if "schedule" in self.dr_params[nonphysical_param] else None
                sched_step = self.dr_params[nonphysical_param]["schedule_steps"] if "schedule" in self.dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = self.dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = self.dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.nonphysical_dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.nonphysical_dr_randomizations[nonphysical_param] = {
                        'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = self.dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = self.dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.nonphysical_dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.nonphysical_dr_randomizations[nonphysical_param] = {
                        'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        # apply randomisations to simulator parameters (e.g. gravity)
        # these are shared across all envs
        if "sim_params" in self.dr_params and do_nonenv_randomize:
            prop_attrs = self.dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)
                }

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop,
                    self.original_props["sim_params"],
                    attr,
                    attr_randomization_params,
                    self.last_step
                )

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise actor properties (hand, object, table, etc)
        # done on a per env basis
        for actor, actor_properties in self.dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():

                    # Randomise the color of all links in an actor
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(
                                env, handle, n, gymapi.MESH_VISUAL,
                                gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                            )
                        continue

                    # Randomise the scale of an actor
                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    # Randomise parameters such as mass, friction, etc
                    # these have property setters/getters
                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

        return randomize_buf
