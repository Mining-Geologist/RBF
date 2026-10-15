"""Deterministic post-cluster oracle-label domain geometry regression."""
import json
import numpy as np
from jizai_labeled_domains import build_labeled_domain_metadata, SPHEROIDAL3_C

def main():
    pts=np.zeros((80,3),dtype=float)
    pts[:,0]=np.arange(80)
    labels=np.ones(80,dtype=int)
    labels[:4]=0
    matrices=np.repeat(np.eye(3)[None,:,:],80,axis=0)
    domains=build_labeled_domain_metadata(pts,labels,matrices,base_range=800.)
    a,b=domains
    assert [a["label"],b["label"]]==[0,1]
    assert a["core_indices"]==[0,1,2,3]
    assert a["support_indices"]==list(range(20))
    assert len(a["support_indices"])==5*len(a["core_indices"])
    assert a["internal_radius"]==17.
    assert abs(a["local_kernel_range"]-17*np.sqrt(SPHEROIDAL3_C))<1e-12
    assert np.allclose(a["bbox_min"],[-17.,-17.,-17.])
    assert np.allclose(a["bbox_max"],[20.,17.,17.])
    assert len(b["support_indices"])==80

    # Verify matrix averaging remains SPD and changes the desired radius by lambda_max.
    aniso=np.repeat(np.diag([.5,1.,2.])[None,:,:],80,axis=0)
    aniso[0]=np.diag([1.,1.,1.])
    modified=build_labeled_domain_metadata(pts,labels,aniso,base_range=40.)
    expected=np.mean(aniso[:4],axis=0)
    assert np.allclose(modified[0]["anisotropy"],expected)
    assert abs(modified[0]["desired_internal_radius"]-40*np.max(np.linalg.eigvalsh(expected))/np.sqrt(SPHEROIDAL3_C))<1e-12

    result={
      "regression":"oracle-label domain metadata; NOT automatic clustering or Leapfrog mesh parity",
      "num_points":len(pts),"num_domains":len(domains),
      "small_domain_core_count":len(a["core_indices"]),
      "small_domain_support_count":len(a["support_indices"]),
      "small_domain_internal_radius":a["internal_radius"],
      "small_domain_local_spheroidal_range":a["local_kernel_range"],
      "support_cap_exact":True,
      "bbox_min":a["bbox_min"],"bbox_max":a["bbox_max"],
      "large_domain_support_count":len(b["support_indices"]),
      "mean_anisotropy_matrix_test":True
    }
    with open("jizai_labeled_domain_test.json","w") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
