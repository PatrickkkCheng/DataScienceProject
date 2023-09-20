from isaacgym import gymapi
from isaacgym import gymutil


def get_sphere_geom(rad=0.01, color=(1, 0, 0)):

    sphere_pose = gymapi.Transform(
        p=gymapi.Vec3(0.0, 0.0, 0.0),
        r=gymapi.Quat(0, 0, 0, 1)
    )

    sphere_geom = gymutil.WireframeSphereGeometry(
        rad,  # rad
        12,  # n_lat
        12,  # n_lon
        sphere_pose,
        color=color
    )
    return sphere_geom
