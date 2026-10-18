"""Model-independent tests for the inferred native inverse snap operation."""
import numpy as np
from invert_native_asphere_snap import inverse_snap, asphere


def main():
    x=np.array([[0.,0.,0.],[10.,0.,0.],[0.,10.,0.]])
    n=np.repeat([[0.,0.,2.]],len(x),axis=0)
    c=np.array([[0.,0.,0.],[10.,10.,0.]])
    coeff=np.array([1.,-.5])
    f=asphere(np.linalg.norm(x[:,None,:]-c[None,:,:],axis=2),10.) @ coeff
    hw=inverse_snap(x,n,c,coeff,'hw',10.)
    fw=inverse_snap(x,n,c,coeff,'fw',10.)
    assert np.allclose(hw[:,2],f)
    assert np.allclose(fw[:,2],-f)
    assert np.allclose(hw[:,:2],x[:,:2])
    assert np.allclose(fw[:,:2],x[:,:2])
    for bad in ('above','below','none'):
        try: inverse_snap(x,n,c,coeff,bad)
        except ValueError:pass
        else: raise AssertionError('bad wall should be rejected')
    print('PASS: saved asphere snap inversion, HW/FW polarity, normalized normals')


if __name__=='__main__':main()
