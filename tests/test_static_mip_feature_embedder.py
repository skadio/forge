from forge.processor import StaticMIPEmbeddings, StaticMIPFeatureEmbedder, _MIPUtils, mip_to_static_embeddings
from tests.test_base import BaseTest

import numpy as np
from unittest.mock import patch


class StaticMIPFeatureEmbedderTest(BaseTest):

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self):
            self.disposed = False

        def dispose(self):
            self.disposed = True

    def test_mip_to_static_embeddings_from_filepath(self):
        embedder = StaticMIPFeatureEmbedder()
        expected = StaticMIPEmbeddings(
            instance_embedding=np.array([1.0], dtype=float),
            embedding_of_constraint=None,
            embedding_of_variable=None,
            instance_feature_names=[],
            constraint_feature_names=[],
            variable_feature_names=[],
            feature_dict={},
        )
        fake_env = StaticMIPFeatureEmbedderTest.FakeEnv()
        fake_model = StaticMIPFeatureEmbedderTest.FakeModel()

        with patch.object(_MIPUtils, "get_mip_items", return_value=["dummy.lp"]) as get_items_mock, \
                patch.object(_MIPUtils, "start_gurobi_env", return_value=fake_env) as start_env_mock, \
                patch("forge.processor.gp.read", return_value=fake_model) as read_mock, \
                patch.object(embedder, "_mip_model_to_embeddings", return_value=expected) as to_embedding_mock:
            out = embedder.mip_to_static_embeddings(
                input_mips="dummy.lp",
                input_mip_instances_file=None,
                has_return=True,
            )

        self.assertIn("dummy.lp", out)
        self.assertIs(out["dummy.lp"], expected)
        self.assertTrue(fake_model.disposed)
        self.assertTrue(fake_env.closed)

        get_items_mock.assert_called_once_with("dummy.lp", None)
        start_env_mock.assert_called_once()
        read_mock.assert_called_once_with("dummy.lp", env=fake_env)
        to_embedding_mock.assert_called_once_with(fake_model)

    def test_convenience_function_delegates(self):
        expected = {
            "dummy.lp": StaticMIPEmbeddings(
                instance_embedding=np.array([1.0], dtype=float),
                embedding_of_constraint=None,
                embedding_of_variable=None,
                instance_feature_names=[],
                constraint_feature_names=[],
                variable_feature_names=[],
                feature_dict={},
            )
        }

        with patch.object(
            StaticMIPFeatureEmbedder,
            "mip_to_static_embeddings",
            return_value=expected,
        ) as delegate_mock:
            out = mip_to_static_embeddings(
                input_mips="dummy.lp",
                input_mip_instances_file=None,
                output_mip_to_embeddings_pkl=None,
                has_return=True,
            )

        self.assertIs(out, expected)
        delegate_mock.assert_called_once_with(
            input_mips="dummy.lp",
            input_mip_instances_file=None,
            output_mip_to_embeddings_pkl=None,
            has_return=True,
        )