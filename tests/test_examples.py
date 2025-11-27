# -*- coding: utf-8 -*-

import random
import logging
import numpy as np
import pandas as pd
from tests.test_base import BaseTest
from forge.embeddings import Forge

class ExampleTest(BaseTest):

    def test_train_quick_start(self):

        # TODO add a quick test
        forge = Forge()
        forge.train()

        self.assertEqual(True, True)