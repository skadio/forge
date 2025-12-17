from forge.labeler import MIPLabeler
from tests.test_base import BaseTest
from forge.utils import Constants
from forge.processor import MIPProcessor


class LabelerTest(BaseTest):

    # TEST main functionality of MIPLabeler
    def test_labeler(self):
        # MIP Labeler
        mip_labeler = MIPLabeler()

        mip_to_gapinfo = mip_labeler.get_mip_to_gapinfo(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                        input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                        output_mip_to_gapinfo_pkl=Constants.default_mip_to_gapinfo_pkl,
                                                        gapinfo_time_limit=10,
                                                        has_return=True)

        mip_files = MIPProcessor.get_only_mip_files(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                    input_mip_instances_file=Constants.default_instances_unit_test_txt)
        self.assertEqual(len(mip_to_gapinfo), len(mip_files))

        for gapinfo in mip_to_gapinfo.values():
            self.assertGreater(gapinfo.gap_ratio, 0.95)
            self.assertLess(gapinfo.gap_ratio, 0.99)
