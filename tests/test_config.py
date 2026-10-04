import unittest
from pathlib import Path
from tunelab.utils.config import load_config


class TestConfigLoading(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent

    def test_base_config_exists_and_loads(self):
        config = load_config(self.root / "configs" / "base.yaml")
        self.assertIn("project_name", config)
        self.assertEqual(config["project_name"], "tunelab")
        self.assertIn("paths", config)
        self.assertIn("evaluation", config)

    def test_dev_config_valid(self):
        config = load_config(self.root / "configs" / "dev.yaml")
        self.assertEqual(config.get("mode"), "dev")
        self.assertEqual(config.get("device"), "cpu")
        self.assertIn("model", config)
        self.assertIn("retrieval", config)

    def test_full_config_valid(self):
        config = load_config(self.root / "configs" / "full.yaml")
        self.assertEqual(config.get("mode"), "full")
        self.assertEqual(config.get("device"), "cuda")
        self.assertIn("training", config)
        self.assertIn("data_fractions", config["training"])


if __name__ == "__main__":
    unittest.main()
