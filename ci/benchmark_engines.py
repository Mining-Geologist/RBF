import json, os, sys, time
import numpy as np

engine = sys.argv[1]
rng = np.random.default_rng(42)
normals = rng.normal(size=(250, 3))
normals /= np.linalg.norm(normals, axis=1)[:, None]
points = 10 * normals

if engine == "jizai":
    import jizai as j
    gen = j.SdfDataGenerator(points, normals)
    interp = j.Interpolant(j.Model(j.Biharmonic3D(dim=3)))
    kwargs = dict(tolerance=1e-5, accuracy=1e-7)
elif engine == "polatory":
    import polatory as p
    import polatory.three as j
    gen = p.SdfDataGenerator(points, normals, 0.1, 1.0)
    interp = j.Interpolant(j.Model(j.Biharmonic3D([])))
    kwargs = dict(tolerance=1e-5, max_iter=100, accuracy=1e-7)
else:
    raise ValueError(engine)

t = time.perf_counter()
interp.fit(gen.sdf_points, gen.sdf_values, **kwargs)
fit_s = time.perf_counter() - t
bbox = j.Bbox(np.full(3, -12.0), np.full(3, 12.0))
field = (j.RbfFieldFunction(interp, accuracy=1e-7) if engine == "jizai" else
         p.RbfFieldFunction(interp, accuracy=1e-7))
iso = (j.Isosurface(bbox, 1.0) if engine == "jizai" else p.Isosurface(bbox, 1.0))
t = time.perf_counter()
mesh = iso.generate_from_seed_points(points, field)
mesh_s = time.perf_counter()-t
mesh.export_obj(f"{engine}_sphere.obj")
out = dict(engine=engine, fit_s=fit_s, mesh_s=mesh_s,
           mesh_bytes=os.path.getsize(f"{engine}_sphere.obj"),
           points=len(points), sdf_points=len(gen.sdf_points))
if hasattr(mesh, "vertices"):
    out["vertices"] = len(mesh.vertices)
if hasattr(mesh, "faces"):
    out["faces"] = len(mesh.faces)
print(json.dumps(out, indent=2))
open(f"{engine}_result.json","w").write(json.dumps(out,indent=2))
