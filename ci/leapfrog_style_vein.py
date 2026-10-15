"""Leapfrog-style implicit vein test using Jizai 1.0.0.

Synthetic, held-out benchmarks: folded mid-surface, hanging-wall/footwall
constraints, contact fit and independent grid accuracy. No LVA claims.
"""
import json
import time
import numpy as np
import jizai as jz

def mid_z(x, y):
    return 0.32*x + 7*np.sin(x/24.0) + 4*np.cos(y/30.0) + 0.002*x*y

def coords(x, y, offset):
    return np.column_stack([x, y, mid_z(x,y)+offset])

rng = np.random.default_rng(2026)
xy = rng.uniform(-65,65,size=(180,2))
# Thin vein: 8 m true vertical separation between walls.
pts = np.concatenate([coords(xy[:,0],xy[:,1],v) for v in (-4.0,0.0,4.0)])
vals = np.concatenate([np.full(len(xy),v) for v in (-4.0,0.0,4.0)])
t=time.perf_counter()
inter=jz.Interpolant(jz.Model(jz.Biharmonic3D(dim=3)))
inter.fit(pts,vals,tolerance=1e-5,accuracy=1e-7)
fit_s=time.perf_counter()-t
held=rng.uniform(-60,60,size=(120,2))
tests={}
for name,zval in [('HW',-4.),('MID',0.),('FW',4.)]:
    pred=np.asarray(inter.evaluate(coords(held[:,0],held[:,1],zval),accuracy=1e-7)).reshape(-1)
    tests[name]={'mae_m':float(np.mean(abs(pred-zval))),
                 'p95_m':float(np.percentile(abs(pred-zval),95)),
                 'signed_bias_m':float(np.mean(pred-zval))}
# Evaluate whether model ordering HW < MID < FW is preserved on held-out traces.
v0=np.asarray(inter.evaluate(coords(held[:,0],held[:,1],-4.),accuracy=1e-7))
v1=np.asarray(inter.evaluate(coords(held[:,0],held[:,1],0.),accuracy=1e-7))
v2=np.asarray(inter.evaluate(coords(held[:,0],held[:,1],4.),accuracy=1e-7))
ordering=float(np.mean((v0<v1)&(v1<v2)))
# Extract the implicit mid-surface using Jizai's surface tracking.
bbox=jz.Bbox(np.array([-75.,-75.,-65.]),np.array([75.,75.,65.]))
field=jz.RbfFieldFunction(inter,accuracy=1e-7)
start=time.perf_counter()
mesh=jz.Isosurface(bbox,3.0).generate_from_seed_points(coords(xy[:,0],xy[:,1],0.0),field)
surface_s=time.perf_counter()-start
mesh.export_obj('jizai_vein_mid.obj')
import os
result={'dataset':'synthetic folded 8-m vein, not Leapfrog reference',
        'training_constraints':len(pts),'holdout_per_contact':len(held),
        'fit_s':fit_s,'surface_s':surface_s,'wall_prediction':tests,
        'ordering_fraction':ordering,'obj_bytes':os.path.getsize('jizai_vein_mid.obj')}
if hasattr(mesh,'vertices'):result['vertices']=len(mesh.vertices)
if hasattr(mesh,'faces'):result['faces']=len(mesh.faces)
print(json.dumps(result,indent=2))
open('jizai_vein_result.json','w').write(json.dumps(result,indent=2))
assert result['obj_bytes']>200
assert ordering>0.95, 'Contact ordering needs investigation'
