from smg_gym.tasks.reorient.smg_reorient import SMGReorient
from smg_gym.tasks.gaiting.smg_gaiting import SMGGaiting
from smg_gym.tasks.gaiting.allegro_gaiting import AllegroGaiting

from smg_gym.tasks.debug.smg_debug import SMGDebug
from smg_gym.tasks.grasping.smg_grasp import SMGGrasp
from smg_gym.tasks.grasping.allegro_grasp import AllegroGrasp

# Mappings from strings to environments
task_map = {
    "smg_reorient": SMGReorient,
    "smg_gaiting": SMGGaiting,
    "smg_debug": SMGDebug,
    "smg_grasp": SMGGrasp,
    "allegro_gaiting": AllegroGaiting,
    "allegro_grasp": AllegroGrasp
}
