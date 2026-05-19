import numpy as np

from forge.embeddings import Forge, MIPEmbeddings
from forge.labeler import HintInfo
from forge.pipeline import mip_to_embeddings, mip_to_hint
from forge.utils import Constants
from tests.test_base import BaseTest


class EmbeddingTest(BaseTest):

    def test_embeddings(self):
        # Forge model with its pre-trained configuration
        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)

        mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                                   input_mips=Constants.DATA_TEST_INSTANCE_DIR,
                                                   input_mip_instances_file=Constants.default_instances_unit_test_txt,
                                                   input_forge_pkl=Constants.default_forge_pretrained_pkl,
                                                   model_type=Constants.FORGE_PRE_TRAIN,
                                                   output_mip_to_embeddings_pkl=Constants.default_mip_to_embeddings_pkl,
                                                   instance_embedding_only=True)

        # print("MIP to Embeddings Dictionary:", mip_to_embeddings_dict)
        with open(Constants.default_instances_unit_test_txt, "r") as f:
            num_instances = sum(1 for line in f if line.strip())
        self.assertEqual(len(mip_to_embeddings_dict), num_instances)

        # ensure each entry is a MIPEmbeddings instance
        for mip, embeddings in mip_to_embeddings_dict.items():
            self.assertIsInstance(embeddings, MIPEmbeddings, f"Embedding for {mip} is not a MIPEmbeddings")

        # # Assert that embeddings are saved and loaded correctly
        # m_to_emb_dict = load_mip_embeddings_hdf5(Constants.default_mip_to_embeddings_pkl)
        # self.assertEqual(len(m_to_emb_dict), len(mip_to_embeddings_dict))
        #
        # # Assert that the instance embeddings match
        # for mip in mip_to_embeddings_dict:
        #     original_embedding = mip_to_embeddings_dict[mip].instance_embedding
        #     loaded_embedding = m_to_emb_dict[mip].instance_embedding
        #     self.assertListAlmostEqual(original_embedding, loaded_embedding)

    def test_variable_proba_hint(self):
        """Test that the variable proba head can be added and hints generated."""
        import gurobipy as gp
        import torch.nn as nn
        from forge.processor import MIPProcessor, _MIPUtils

        forge = Forge(train_config_yaml=Constants.default_train_config_yaml)

        # Load pretrained model with variable proba head
        forge.load_model(input_forge_pkl=Constants.default_forge_pretrained_pkl,
                         model_type=Constants.FORGE_FINE_TUNE_VARIABLE_PROBA)

        # Manually create the variable_proba_layer since load_model only sets the flag
        forge.variable_proba_layer = nn.Linear(forge.updated_input_dim, 1).to(forge.device)

        # Load a MIP instance
        mip_files = _MIPUtils.get_only_mip_files(
            input_mip_folder=Constants.DATA_TEST_INSTANCE_DIR,
            input_mip_instances_file=Constants.default_instances_unit_test_txt,
            is_sort_by_size=False
        )
        gurobi_env = _MIPUtils.start_gurobi_env()
        mip_model = gp.read(mip_files[0], env=gurobi_env)

        hint_info = forge._mip_model_to_hint(mip_model, prob_type='SC')

        gurobi_env.close()

        self.assertIsInstance(hint_info, HintInfo)
        self.assertIsInstance(hint_info.hint_ones, type(np.array([])))
        self.assertIsInstance(hint_info.hint_zeros, type(np.array([])))
        self.assertIsInstance(hint_info.hint_pri_ones, list)
        self.assertIsInstance(hint_info.hint_pri_zeros, list)
