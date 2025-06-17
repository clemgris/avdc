# === Standard Library ===
import logging
import sys
from pathlib import Path

# === Third-party Libraries ===
import torchvision

# === Project Path Setup ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Imports ===
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel

logger = logging.getLogger(__name__)


class ExpertModel(CalvinBaseModel):
    def __init__(
        self,
    ):
        self.reset()

    def reset(self):
        """
        This is called
        """
        self.steps = 0

    def save_image(self, image, name):
        saving_path = Path(self.debug_path) / name
        torchvision.utils.save_image((image + 1) / 2, saving_path)

    def step(self, obs, text_goal, oracle_subgoals=None):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        self.actions = oracle_subgoals["actions"]
        action_idx = min(self.steps, len(self.actions) - 1)

        selected_action = self.actions[action_idx]
        self.steps += 1
        return selected_action
