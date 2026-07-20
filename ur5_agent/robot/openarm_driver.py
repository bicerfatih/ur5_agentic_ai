# robot/openarm_driver.py — placeholder for airport OpenArm integration

from robot.base import RobotDriver
from robot.mock_driver import MockDriver


class OpenArmDriver(RobotDriver):
    """
    Future driver for OpenArm robots in airport operations.
    Hardware SDK not wired yet — use --dry-run or MockDriver until integration.
    """

    def __init__(self, host: str = "", dry_run: bool = False):
        self.host = host
        self._dry_run = dry_run
        self._mock: MockDriver | None = None

    @property
    def arm_model(self) -> str:
        return "openarm"

    @property
    def is_simulated(self) -> bool:
        return self._dry_run or self._mock is not None

    def connect(self):
        if self._dry_run:
            self._mock = MockDriver(arm_model="openarm", label="openarm_stub")
            self._mock.connect()
            print(
                "[OpenArm] Dry-run stub active. "
                "Replace openarm_driver.py with vendor SDK when hardware is available.\n"
            )
            return
        raise ConnectionError(
            "OpenArm hardware driver not implemented yet. "
            "Use --robot ur5 for physical tests or --robot openarm --dry-run to develop agent flows."
        )

    def disconnect(self):
        if self._mock:
            self._mock.disconnect()
            self._mock = None

    def is_connected(self) -> bool:
        return self._mock is not None and self._mock.is_connected()

    def _delegate(self) -> RobotDriver:
        if not self._mock:
            raise RuntimeError("OpenArm not connected")
        return self._mock

    def get_joint_positions(self) -> list:
        return self._delegate().get_joint_positions()

    def get_tcp_pose(self) -> list:
        return self._delegate().get_tcp_pose()

    def get_tcp_force(self) -> list:
        return self._delegate().get_tcp_force()

    def get_robot_mode(self) -> int:
        return self._delegate().get_robot_mode()

    def get_safety_mode(self) -> int:
        return self._delegate().get_safety_mode()

    def get_full_state(self) -> dict:
        state = self._delegate().get_full_state()
        state["arm_model"] = "openarm"
        state["integration"] = "stub"
        return state

    def move_joint(self, joints: list, speed: float = 0.3, accel: float = 0.3):
        self._delegate().move_joint(joints, speed, accel)

    def move_linear(self, tcp_pose: list, speed: float = 0.1, accel: float = 0.1):
        self._delegate().move_linear(tcp_pose, speed, accel)

    def move_home(self):
        self._delegate().move_home()

    def stop(self):
        self._delegate().stop()

    def move_tcp_relative(
        self, dx=0.0, dy=0.0, dz=0.0, speed: float = 0.1, accel: float = 0.1
    ):
        self._delegate().move_tcp_relative(dx, dy, dz, speed, accel)
