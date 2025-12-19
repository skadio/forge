from forge.embeddings import Forge
from forge.pipeline import mip_to_mipinfo
from tests.test_base import BaseTest

from forge.utils import Constants


class MipInfoTest(BaseTest):

    def test_mipinfo(self):
        # Forge model
        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)

        # Pretrain forge
        relaxation_list = [0.05, 0.01]
        mip_to_mipinfo_dict = mip_to_mipinfo(forge,
                                             input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                             input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                             relaxation_list=relaxation_list,
                                             output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl)

        with open(Constants.default_instances_unit_test_txt, "r") as f:
            num_instances = sum(1 for line in f if line.strip())
        self.assertEqual(len(mip_to_mipinfo_dict), num_instances * (len(relaxation_list) + 1))