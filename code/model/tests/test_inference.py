"""Check sampler against a known bounded Gaussian posterior, not flight code."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import truncnorm
from inference import posterior
from physics import LOW,HIGH


class InferenceTests(unittest.TestCase):
    def test_known_truncated_gaussian(self):
        target=np.array([27.,2.,6.]); sigma=3.
        def linear_forward(theta,cfg):
            n=len(theta)
            return dict(observations=np.repeat(theta[:,:3,None],2,axis=2).transpose(0,2,1),
                endpoint=theta[:,:2],status=np.ones(n,dtype=int))
        with patch('inference.simulate',linear_forward):
            r=posterior(n=12000,sigma=sigma,seed=12,observations=np.tile(target,(2,1)))
        sd=sigma/np.sqrt(2)
        expected=truncnorm.mean((LOW[:2]-target[:2])/sd,(HIGH[:2]-target[:2])/sd,loc=target[:2],scale=sd)
        np.testing.assert_allclose(r['conditional_mean'],expected,atol=.16)
        self.assertGreater(r['conditional_ess'],500)


if __name__=='__main__':unittest.main()
