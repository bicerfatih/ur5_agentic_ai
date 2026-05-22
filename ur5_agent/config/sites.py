# config/sites.py — deployment profiles (lab → airport / Emirates)

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteProfile:
    site_id: str
    display_name: str
    environment: str  # lab | airport_ground | airport_cargo
    operator: str  # dev | emirates
    max_joint_speed: float
    max_linear_speed: float
    max_single_move_down: float
    max_steps_per_goal: int
    require_state_before_move: bool
    human_proximity_strict: bool
    allowed_arm_models: tuple[str, ...]


SITES: dict[str, SiteProfile] = {
    "lab": SiteProfile(
        site_id="lab",
        display_name="Development Lab",
        environment="lab",
        operator="dev",
        max_joint_speed=0.5,
        max_linear_speed=0.2,
        max_single_move_down=0.10,
        max_steps_per_goal=20,
        require_state_before_move=True,
        human_proximity_strict=False,
        allowed_arm_models=("ur5", "ur5e", "ur", "openarm"),
    ),
    "airport_ground": SiteProfile(
        site_id="airport_ground",
        display_name="Airport Ground Operations",
        environment="airport_ground",
        operator="emirates",
        max_joint_speed=0.25,
        max_linear_speed=0.08,
        max_single_move_down=0.05,
        max_steps_per_goal=15,
        require_state_before_move=True,
        human_proximity_strict=True,
        allowed_arm_models=("openarm", "ur5"),
    ),
    "airport_cargo": SiteProfile(
        site_id="airport_cargo",
        display_name="Airport Cargo / Baggage",
        environment="airport_cargo",
        operator="emirates",
        max_joint_speed=0.35,
        max_linear_speed=0.12,
        max_single_move_down=0.08,
        max_steps_per_goal=18,
        require_state_before_move=True,
        human_proximity_strict=True,
        allowed_arm_models=("openarm", "ur5"),
    ),
}

DEFAULT_SITE = "lab"


def get_site(site_id: str | None) -> SiteProfile:
    key = (site_id or DEFAULT_SITE).lower()
    if key not in SITES:
        raise ValueError(
            f"Unknown site '{site_id}'. Choose from: {', '.join(SITES.keys())}"
        )
    return SITES[key]
