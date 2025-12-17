from forge.embeddings import Forge, MIPEmbeddings
from forge.pipeline import mip_to_embeddings
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
                                                   output_mip_to_embeddings_pkl=Constants.default_mip_to_embeddings_pkl)

        # print("MIP to Embeddings Dictionary:", mip_to_embeddings_dict)
        with open(Constants.default_instances_unit_test_txt, "r") as f:
            num_instances = sum(1 for line in f if line.strip())
        self.assertEqual(len(mip_to_embeddings_dict), num_instances)

        # ensure each entry is a MIPEmbeddings instance
        for mip, embeddings in mip_to_embeddings_dict.items():
            self.assertIsInstance(embeddings, MIPEmbeddings, f"Embedding for {mip} is not a MIPEmbeddings")
