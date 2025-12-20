from forge.processor import _MIPUtils
from tests.test_base import BaseTest
import tempfile
from pathlib import Path
from unittest.mock import patch


class MIPUtilsTest(BaseTest):
    class DummyModel:
        pass

    def test_single_file_path(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "instance.mps"
            p.write_text("")  # create empty file
            out = _MIPUtils.get_mip_items(str(p), input_mip_instances_file=None)
            self.assertEqual(out, [str(p)])

    def test_directory_returns_only_mip_files(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.mps"
            p2 = Path(td) / "b.lp"
            p3 = Path(td) / "ignore.txt"
            p1.write_text("")
            p2.write_text("")
            p3.write_text("")
            out = _MIPUtils.get_mip_items(str(td), input_mip_instances_file=None)
            self.assertEqual(set(out), {str(p1), str(p2)})

    def test_single_model_instance(self):
        with patch("forge.processor.gp.Model", MIPUtilsTest.DummyModel):
            model = MIPUtilsTest.DummyModel()
            out = _MIPUtils.get_mip_items(model, input_mip_instances_file=None)
            self.assertEqual(out, [model])

    def test_list_with_model_and_file(self):
        with tempfile.TemporaryDirectory() as td, patch("forge.processor.gp.Model", MIPUtilsTest.DummyModel):
            p = Path(td) / "f.mps"
            p.write_text("")
            model = MIPUtilsTest.DummyModel()
            out = _MIPUtils.get_mip_items([model, str(p)], input_mip_instances_file=None)
            # first item should be the model object, second (or present) should be the path
            self.assertIn(model, out)
            self.assertIn(str(p), out)

    def test_nonexistent_path_raises_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            fake = str(Path(td) / "does_not_exist.mps")
            with self.assertRaises(ValueError):
                _MIPUtils.get_mip_items(fake, input_mip_instances_file=None)
