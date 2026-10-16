"""Native-free controls for the experimental real-point domain graph."""
import numpy as np
from real_point_domain_graph import Configuration, cluster_real_points

def run():
    rng = np.random.default_rng(2026)
    pts = np.r_[rng.normal(-10, 1, (20,3)), rng.normal(10,1,(20,3))]
    tensors = np.repeat(np.eye(3)[None], len(pts), axis=0)
    for cfg in (Configuration(neighbours=6,max_fraction=.10),
                Configuration(neighbours=12,max_fraction=.50),
                Configuration(neighbours=32,max_fraction=.15)):
        a,info=cluster_real_points(pts,tensors,cfg)
        b,_=cluster_real_points(pts,tensors,cfg)
        assert np.array_equal(a,b)
        assert len(a)==len(pts)
        assert len(np.unique(a))==info["domain_count"]
        assert np.bincount(a).max() <= max(cfg.micro_size,int(np.floor(cfg.max_fraction*len(pts))))
        assert info["micro_merges"] + info["macro_merges"] == len(pts)-info["domain_count"]
    try:
        cluster_real_points(pts,tensors[:20],Configuration())
    except ValueError: pass
    else: raise AssertionError("Expected shape error")
    print("PASS: offline region growth, size caps, merge accounting, deterministic labels")

if __name__=="__main__":
    run()
