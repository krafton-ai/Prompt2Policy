"""Shadow Hand + Pen environment for pen spinning tasks.

Registers ``Isaac-Spin-Pen-Shadow-Direct-v0`` with Gymnasium.
Uses the same ``InHandManipulationEnv`` as the cube task but swaps the
DexCube object for a cylinder primitive (pen stand-in).

Anyone who pulls SAR and has IsaacLab installed gets this env automatically
-- no modifications to IsaacLab source required.
"""

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

# ---------------------------------------------------------------------------
# Pen dimensions (approximate ballpoint pen)
# ---------------------------------------------------------------------------
PEN_RADIUS = 0.009  # 9 mm radius (~18 mm diameter)
PEN_HEIGHT = 0.14  # 140 mm length


@configclass
class ShadowHandPenEnvCfg(ShadowHandEnvCfg):
    """Shadow Hand environment with a pen-shaped object instead of a cube.

    Inherits all hand config, scene config, observation layout, and domain
    randomization from the upstream ``ShadowHandEnvCfg``.  Only the
    manipulated object and goal marker are replaced.
    """

    # Override: pen object (cylinder primitive)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CylinderCfg(
            radius=PEN_RADIUS,
            height=PEN_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # ~32 g at these dimensions (real pen ~12 g); heavier to aid contact
            mass_props=sim_utils.MassPropertiesCfg(density=900.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.39, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Override: goal marker matches pen shape
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=PEN_RADIUS,
                height=PEN_HEIGHT,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    opacity=0.5,
                ),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

_ENTRY_POINT = (
    "isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv"
)

gym.register(
    id="Isaac-Spin-Pen-Shadow-Direct-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ShadowHandPenEnvCfg",
    },
)
