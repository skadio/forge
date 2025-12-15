from forge.processor import MIPProcessor
from tests.test_base import BaseTest
from forge.utils import Constants
import tempfile
from pathlib import Path
from unittest.mock import patch


class ProcessorTest(BaseTest):
    class DummyModel:
        pass

    # TEST main functionality of MIPProcessor
    def test_processor(self):
        # MIP Processor
        mip_proc = MIPProcessor()

        # Create MIP to MIPInfo dictionary
        relaxation_list = [0.05, 0.01]
        mip_to_mipinfo = mip_proc.convert_mip_to_mipinfo(input_mip_folder=Constants.DATA_TESTS_DIR,
                                                         output_mip_to_mipinfo_pkl=Constants.default_mip_to_mipinfo_pkl,
                                                         relaxation_list=relaxation_list,
                                                         has_return=True)

        # List of MIPInfo objects for training
        mipinfo_list = mip_proc.load_mipinfo_from_pickles([Constants.default_mip_to_mipinfo_pkl])

        # We create one relaxed instance per relaxation ratio for each MIP instance
        # E.g., if there are 5 MIP instances and 2 relaxation ratios, we should have 15 MIPInfo objects
        input_folder = Path(Constants.DATA_TESTS_DIR)
        num_instances = sum(1 for p in input_folder.iterdir() if p.is_file())
        self.assertEqual(len(mipinfo_list), num_instances * (len(relaxation_list) + 1))

    # TEST MIPProcessor.get_mip_items()
    def test_single_file_path(self):
        mip_proc = MIPProcessor()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "instance.mps"
            p.write_text("")  # create empty file
            out = mip_proc.get_mip_items(str(p))
            self.assertEqual(out, [str(p)])

    # TEST MIPProcessor.get_mip_items()
    def test_directory_returns_only_mip_files(self):
        mip_proc = MIPProcessor()
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.mps"
            p2 = Path(td) / "b.lp"
            p3 = Path(td) / "ignore.txt"
            p1.write_text("")
            p2.write_text("")
            p3.write_text("")
            out = mip_proc.get_mip_items(str(td))
            self.assertEqual(set(out), {str(p1), str(p2)})

    # TEST MIPProcessor.get_mip_items()
    def test_single_model_instance(self):
        mip_proc = MIPProcessor()
        with patch("forge.processor.gp.Model", ProcessorTest.DummyModel):
            model = ProcessorTest.DummyModel()
            out = mip_proc.get_mip_items(model)
            self.assertEqual(out, [model])

    # TEST MIPProcessor.get_mip_items()
    def test_list_with_model_and_file(self):
        mip_proc = MIPProcessor()
        with tempfile.TemporaryDirectory() as td, patch("forge.processor.gp.Model", ProcessorTest.DummyModel):
            p = Path(td) / "f.mps"
            p.write_text("")
            model = ProcessorTest.DummyModel()
            out = mip_proc.get_mip_items([model, str(p)])
            # first item should be the model object, second (or present) should be the path
            self.assertIn(model, out)
            self.assertIn(str(p), out)

    # TEST MIPProcessor.get_mip_items()
    def test_nonexistent_path_raises_value_error(self):
        mip_proc = MIPProcessor()
        with tempfile.TemporaryDirectory() as td:
            fake = str(Path(td) / "does_not_exist.mps")
            with self.assertRaises(ValueError):
                mip_proc.get_mip_items(fake)
