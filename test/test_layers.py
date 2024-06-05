import unittest
import numpy as np

from ou_dl.Layers import Dnn
from ou_dl.Module import seq

class test_oudl(unittest.TestCase):
    def test_Dnn(self):
        ind, outd = np.random.randint(1, 1000, 2)
        single_dnn = Dnn(ind,outd,)

        self.assertEqual(single_dnn.weight.shape, (ind,outd))
        self.assertEqual(single_dnn.bias.shape, (outd,))
        self.assertEqual(single_dnn(np.random.randn(100,ind)).shape, (100,outd,) )

    def test_seq(self):
        ind, midd, outd = np.random.randint(1, 1000, 3)
        ind = 1
        outd = 1
        seq_dnn = seq(
            Dnn(ind, midd,), 
            Dnn(midd, outd,),
            )
        print(seq_dnn)
        self.assertEqual(seq_dnn(np.random.randn(ind)).shape, (outd, ) )

        