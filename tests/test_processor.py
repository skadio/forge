from forge.processor import MIPProcessor, _MIPUtils
from tests.test_base import BaseTest
from forge.utils import Constants


class ProcessorTest(BaseTest):

    def test_processor_single_thread(self):
        # MIP Processor
        mip_proc = MIPProcessor()

        # Create MIP to MIPInfo dictionary
        relaxation_list = [0.05, 0.01]
        mip_to_mipinfo = mip_proc.convert_mip_to_mipinfo(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                         input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                         output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                                                         relaxation_list=relaxation_list,
                                                         num_parallel_workers=1,
                                                         has_return=True)

        # List of MIPInfo objects for training
        mipinfo_list = _MIPUtils.load_mipinfo_from_pickles([Constants.default_mip_to_mipinfo_pkl])

        # We create one relaxed instance per relaxation ratio for each MIP instance
        # E.g., if there are 5 MIP instances and 2 relaxation ratios, we should have 15 MIPInfo objects
        with open(Constants.default_instances_unit_test_txt, "r") as f:
            num_instances = sum(1 for line in f if line.strip())
        self.assertEqual(len(mipinfo_list), num_instances * (len(relaxation_list) + 1))

    def test_processor_multi_thread(self):
        # MIP Processor
        mip_proc = MIPProcessor()

        # Create MIP to MIPInfo dictionary
        relaxation_list = [0.05, 0.01]
        mip_to_mipinfo = mip_proc.convert_mip_to_mipinfo(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                         input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                         output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                                                         relaxation_list=relaxation_list,
                                                         num_parallel_workers=3,
                                                         has_return=True)

        # List of MIPInfo objects for training
        mipinfo_list = _MIPUtils.load_mipinfo_from_pickles([Constants.default_mip_to_mipinfo_pkl])

        # We create one relaxed instance per relaxation ratio for each MIP instance
        # E.g., if there are 5 MIP instances and 2 relaxation ratios, we should have 15 MIPInfo objects
        with open(Constants.default_instances_unit_test_txt, "r") as f:
            num_instances = sum(1 for line in f if line.strip())
        self.assertEqual(len(mipinfo_list), num_instances * (len(relaxation_list) + 1))
