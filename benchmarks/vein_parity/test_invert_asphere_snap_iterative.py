"""Native-independent exact recovery checks for iterative asphere wall inversion."""
import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from invert_native_asphere_snap import asphere
from invert_asphere_snap_iterative import inverse_iterative

def run():
    xs=np.linspace(-3,3,7);ys=np.linspace(-3,3,7)
    grid=np.stack(np.meshgrid(xs,ys,indexing='ij'),axis=-1)
    base=np.column_stack([grid.reshape(-1,2),np.zeros(grid.shape[0]*grid.shape[1])])
    faces=[]
    for i in range(6):
        for j in range(6):
            a=i*7+j;b=a+7
            faces.extend([[a,b,a+1],[b,b+1,a+1]])
    faces=np.array(faces,int)
    n=trimesh.Trimesh(vertices=base,faces=faces,process=False).vertex_normals
    centres=np.array([[0.,0.,0.],[2.,2.,0.]])
    coeff=np.array([0.15,-0.05])
    f=asphere(cdist(base,centres),10.)@coeff
    for side,polarity in [('hw',1),('fw',-1)]:
        # Known test forward deformation, evaluated on base coordinates.
        final=base-polarity*n*f[:,None]
        x,meta=inverse_iterative(final,faces,centres,coeff,side,
                                 scale=10.,damping=.65,iterations=30)
        error=np.sqrt(np.mean(np.sum((x-base)**2,axis=1)))
        assert error<.003,(side,error,meta['best_iteration'])
        assert meta['min_internal_fixed_point_residual'] < .001
    try:
        inverse_iterative(base,faces,centres,coeff,'other')
    except ValueError:pass
    else:raise AssertionError('Must reject incorrect HW/FW side')
    print('PASS: iterative inverse recovers both HW/FW synthetic surfaces without oracle stopping')

if __name__=='__main__':run()
