#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
#sys.path.insert(1, os.path.join('..', os.getcwd()))
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

## import model specific functions and variables
from model import *



# call the test from the main directory
# otherwise model and logs are stored in wrong place or it can not find directories
class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        data_dir = os.path.join("data","cs-train")
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test-united_kingdom-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """

        ## train the model
        all_data, all_models = model_load()
        model = all_models['united_kingdom']
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))


    def test_03_predict(self):
        """
        test the predict function input
        """

        ## ensure that a list can be passed
        result = model_predict('united_kingdom','2019','01','01',test=True)
        y_pred = result['y_pred']
        # Since revenue is predicted this can be 0 or higher
        self.assertTrue(y_pred >= 0)


### Run the tests
if __name__ == '__main__':
    unittest.main()
