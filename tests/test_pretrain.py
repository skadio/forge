from forge.embeddings import Forge
from forge.pipeline import pretrain
from tests.test_base import BaseTest

from forge.utils import Constants


class PretrainTest(BaseTest):

    def test_pretrain(self):
        # Forge model
        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)

        # Pretrain forge
        pretrain(forge,
                 input_mip_folder=Constants.DATA_TEST_DIR,
                 relaxation_list=[0.05, 0.01],
                 output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                 output_forge_pretrained_pkl=Constants.default_forge_pretrained_pkl,
                 output_log_file=Constants.default_forge_log_file)

        self.assertEqual(True, True)
