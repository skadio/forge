import os
import subprocess
import sys
from unittest import skip

from forge.utils import Constants
from tests.test_base import BaseTest


class ReadmeTest(BaseTest):

    @skip("Skipping command line test to speed up CI")
    def test_pretrain(self):
        from forge.embeddings import Forge
        from forge.pipeline import pretrain

        # Forge model with its pre-training configuration
        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)

        # Pretrain Forge on a set of MIP instances
        pretrain(forge=forge,
                 input_mip_folder=Constants.DATA_TRAINING_DIR,
                 input_mip_instances_file=Constants.default_instances_unit_test_txt,
                 output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                 output_forge_pretrained_pkl=Constants.default_forge_pretrained_pkl,
                 output_log_file=Constants.default_forge_log_file,
                 epochs=1,
                 steps_per_instance=1)


    @skip("Skipping command line test to speed up CI")
    def test_pretrain_command(self):
        # project root is one level above tests/
        project_root = os.path.dirname(os.path.dirname(__file__))

        cmd = [
            sys.executable,
            "-m",
            "scripts.pretrain",
            "--train_config_yaml", os.path.join(project_root, "forge", "configs", "train_config.yaml"),
            "--input_mip_folder", os.path.join(project_root, "data", "instances"),
            "--input_mip_instances_file", os.path.join(project_root, "data", "configs", "pretrain.txt"),
            "--output_mip_to_mipinfo_pkl", os.path.join(project_root, "models", "mip_to_mipinfo.pkl"),
            "--output_forge_pretrained_pkl", os.path.join(project_root, "models", "forge_pretrained.pkl"),
            "--output_log_file", os.path.join(project_root, "models", "forge_pretrained.log"),
        ]

        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        assert result.returncode == 0, f"pretrain failed: {result.stderr}"