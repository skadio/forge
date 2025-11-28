from forge.processor import MIPProcessor
from tests.test_base import BaseTest
from forge.utils import Constants


class ProcessorTest(BaseTest):

    def test_processor(self):
        # MIP Processor
        mip_processor = MIPProcessor()

        # Create MIP to MIPInfo dictionary
        mip_to_mipinfo = mip_processor.convert_mip_to_mipinfo(input_mip_folder=Constants.DATA_TEST_DIR,
                                                              output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                                                              relaxation_list=[0.05, 0.01],
                                                              has_return=True)

        # List of MIPInfo objects for training
        mipinfo_list = mip_processor.load_mipinfo_from_pickles([Constants.default_mip_to_mipinfo_pkl])
