from forge.labeler import MIPLabeler
from forge.embeddings import Forge
from tests.test_base import BaseTest
from forge.utils import Constants
from forge.processor import _MIPUtils


class LabelerTest(BaseTest):

    def test_labeler_single_thread(self):
        # MIP Labeler
        mip_labeler = MIPLabeler()

        mip_to_gapinfo = mip_labeler.convert_mip_to_gapinfo(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                            input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                            output_mip_to_gapinfo_pkl=Constants.default_mip_to_gapinfo_pkl,
                                                            gapinfo_time_limit=10,
                                                            num_parallel_workers=1,
                                                            has_return=True)

        mip_files = _MIPUtils.get_only_mip_files(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                 input_mip_instances_file=Constants.default_instances_unit_test_txt)
        self.assertEqual(len(mip_to_gapinfo), len(mip_files))

        for gapinfo in mip_to_gapinfo.values():
            self.assertGreater(gapinfo.gap_ratio, 0.95)
            self.assertLess(gapinfo.gap_ratio, 0.99)

    def test_labeler_multi_thread(self):
        # MIP Labeler
        mip_labeler = MIPLabeler()

        mip_to_gapinfo = mip_labeler.convert_mip_to_gapinfo(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                            input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                            output_mip_to_gapinfo_pkl=Constants.default_mip_to_gapinfo_pkl,
                                                            gapinfo_time_limit=10,
                                                            num_parallel_workers=5,
                                                            has_return=True)

        mip_files = _MIPUtils.get_only_mip_files(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                 input_mip_instances_file=Constants.default_instances_unit_test_txt)
        self.assertEqual(len(mip_to_gapinfo), len(mip_files))

        for gapinfo in mip_to_gapinfo.values():
            self.assertGreater(gapinfo.gap_ratio, 0.95)
            self.assertLess(gapinfo.gap_ratio, 0.99)

    def test_labeler_triplet(self):
        # Load pretrained Forge model for generating triplet labels
        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)
        forge.load_model(input_forge_pkl=Constants.default_forge_pretrained_pkl,
                         model_type=Constants.FORGE_PRE_TRAIN)

        # MIP Labeler
        mip_labeler = MIPLabeler()

        mip_to_tripletinfo = mip_labeler.convert_mip_to_tripletinfo(
            input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
            input_mip_instances_file=Constants.default_instances_unit_test_txt,
            output_mip_to_tripletinfo_pkl=Constants.default_mip_to_tripletinfo_pkl,
            forge_model=forge,
            triplet_time_limit=10,
            triplet_num_solutions=3,
            has_return=True
        )

        mip_files = _MIPUtils.get_only_mip_files(input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
                                                 input_mip_instances_file=Constants.default_instances_unit_test_txt)

        # Not all instances may produce triplets (depends on solution count)
        for mip_file in mip_files:
            if mip_file in mip_to_tripletinfo:
                triplet_info = mip_to_tripletinfo[mip_file]
                # Triplets should have shape (N, 3)
                self.assertEqual(triplet_info.triplets.ndim, 2)
                self.assertEqual(triplet_info.triplets.shape[1], 3)
                # y_true should be a 2D tensor with shape (num_vars, 1)
                self.assertEqual(triplet_info.y_true.ndim, 2)
                self.assertEqual(triplet_info.y_true.shape[1], 1)
                # At least some variables should appear in solutions
                self.assertGreater(triplet_info.y_true.sum().item(), 0)
