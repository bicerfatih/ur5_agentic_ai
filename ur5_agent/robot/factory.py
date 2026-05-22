# robot/factory.py — create the right driver from CLI / env

import os

from robot.base import RobotDriver
from robot.mock_driver import MockDriver
from robot.openarm_driver import OpenArmDriver
from robot.ur5_driver import UR5Driver


def create_robot(
    robot_type: str = "ur5",
    dry_run: bool = False,
    host: str | None = None,
) -> RobotDriver:
    robot_type = (robot_type or os.environ.get("ROBOT_TYPE", "ur5")).lower()
    dry_run = dry_run or os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

    if dry_run:
        label = f"{robot_type}_dry_run"
        return MockDriver(arm_model=robot_type, label=label)

    if robot_type in ("ur5", "ur5e", "ur"):
        from config.settings import ROBOT_HOST as default_host

        return UR5Driver(host=host or os.environ.get("ROBOT_HOST") or default_host)

    if robot_type == "openarm":
        return OpenArmDriver(
            host=host or os.environ.get("ROBOT_HOST", ""),
            dry_run=False,
        )

    raise ValueError(f"Unknown robot type: {robot_type}. Use ur5 or openarm.")
