# -*- coding: utf-8 -*-

from forge.embeddings import Forge
from tests.test_base import BaseTest


class ExampleTest(BaseTest):

    def test_train_quick_start(self):

        # TODO add a quick test
        forge = Forge()
        forge.train()

        self.assertEqual(True, True)