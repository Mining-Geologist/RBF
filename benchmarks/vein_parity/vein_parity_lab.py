"""Independent, reference-coordinate vein/pinchout proof of concept.

Evidence-based behaviours are marked in the docs. Interpolation, reversal offset,
clustering of contacts and meshing are ORIGINAL TESTABLE HYPOTHESES and are not
claimed to be identical to proprietary Leapfrog implementation.

Requirements: numpy, scipy. Optional: skimage, trimesh for a closed OBJ solid.
No Leapfrog executable, project reader or licence service is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    code: str


@dataclass
class Hole:
    name: str
    path: np.ndarray  # rows: measured depth, X, Y, Z
    logs: list[Interval] = field(default_factory=list)

    def __post_init__(self):
        self.path = np.asarray(self.path, float)
        if self.path.ndim != 2 or self.path.shape[1] != 4 or len(self.path) < 2:
            raise ValueError("Hole path must be (N,4): measured depth and XYZ")
        if not np.isfinite(self.path).all() or np.any(np.diff(self.path[:, 0]) <= 0):
            raise ValueError("Measured depths must be increasing and finite")
        self.logs.sort(key=lambda x: x.start)
        for i, item in enumerate(self.logs):
            if item.end <= item.start or item.start < self.path[0, 0]-1e-8 or item.end > self.path[-1, 0]+1e-8:
                raise ValueError("Invalid interval or outside the hole")
            if i and item.start < self.logs[i-1].end-1e-8:
                raise ValueError("Overlapping lithology intervals")

    def at(self, md):
        return np.array([np.interp(md, self.path[:, 0], self.path[:, k]) for k in (1,2,3)])


@dataclass
class Frame:
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    n_axis: np.ndarray

    @classmethod
    def from_normal(cls, origin, normal):
        n = np.asarray(normal, float)
        n /= np.linalg.norm(n)
        up = np.array([0., 0., 1.])
        trial = up if abs(np.dot(n, up)) < .85 else np.array([0., 1., 0.])
        u = np.cross(trial, n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return cls(np.asarray(origin, float), u, v, n)

    def to_local(self, xyz):
        d = np.asarray(xyz, float)-self.origin
        return np.stack([d @ self.u_axis, d @ self.v_axis, d @ self.n_axis], axis=-1)

    def to_global(self, uvw):
        q = np.asarray(uvw, float)
        return (self.origin + q[..., 0, None]*self.u_axis
                + q[..., 1, None]*self.v_axis + q[..., 2, None]*self.n_axis)


@dataclass(frozen=True)
class Options:
    vein_code: str = "VEIN"
    pinch_out: bool = False
    pinch_empty_traces: bool = False
    dont_pinch_far_logged: bool = False
    logged_distance: float = 5.
    excluded_pinch_holes: frozenset[str] = frozenset()
    include_end_wall_points: bool = True
    include_end_midpoints: bool = False
    multiple_intersections: bool = True
    reference: str = "planar"  # planar or curved
    min_thickness: Optional[float] = None
    max_thickness: Optional[float] = None
    smoothing: float = .02
    reversal_halfwidth: Optional[float] = None

    def __post_init__(self):
        if self.pinch_out and self.min_thickness is not None:
            raise ValueError("Minimum thickness is incompatible with pinchout")
        if self.reference not in ("planar", "curved"):
            raise ValueError("Only planar and simple curved reference currently supported")
        if self.min_thickness is not None and self.min_thickness < 0:
            raise ValueError("Minimum thickness cannot be negative")
        if self.max_thickness is not None and self.max_thickness <= 0:
            raise ValueError("Maximum thickness must be positive")
        if self.logged_distance < 0 or self.smoothing < 0:
            raise ValueError("Distances/smoothing must be nonnegative")


@dataclass
class Contact:
    hole: str
    md: float
    xyz: np.ndarray
    wall: str
    source: str  # observed or reversed-outside or end


@dataclass
class Midpoint:
    hole: str
    md: float
    xyz: np.ndarray
    included: bool


@dataclass
class PinchCandidate:
    hole: str
    decision: str
    md: Optional[float]
    distance_to_logged_interval: Optional[float]


class Interpolant:
    """Smooth scalar field in reference (u,v); never an undocumented native RBF."""
    def __init__(self, uv, vals, smoothing=.02):
        uv = np.asarray(uv, float)
        vals = np.asarray(vals, float)
        if uv.ndim != 2 or uv.shape[1] != 2 or vals.shape != (len(uv),):
            raise ValueError("Expected points Nx2, scalar values N")
        # Merge exact duplicate (u,v) locations before interpolation.
        rounded = np.round(uv, 9)
        _, reverse = np.unique(rounded, axis=0, return_inverse=True)
        x = np.zeros((reverse.max()+1, 2))
        y = np.zeros(len(x))
        cnt = np.bincount(reverse)
        np.add.at(x, reverse, uv)
        np.add.at(y, reverse, vals)
        x /= cnt[:, None]
        y /= cnt
        self.constant = float(np.mean(y))
        self.rbf = None
        if len(x) >= 3 and np.linalg.matrix_rank(x-x.mean(axis=0)) == 2:
            self.rbf = RBFInterpolator(x, y, kernel="thin_plate_spline", smoothing=smoothing, degree=1)
        elif len(x) >= 2:
            self.rbf = RBFInterpolator(x, y, kernel="linear", smoothing=max(smoothing, 1e-9), degree=-1)

    def __call__(self, uv):
        uv = np.asarray(uv, float)
        shape = uv.shape[:-1]
        if self.rbf is None:
            return np.full(shape, self.constant)
        return self.rbf(uv.reshape(-1, 2)).reshape(shape)


class VeinModel:
    """HW/FW scalar offset fields, pinchout decisions and geological solid mask."""
    def __init__(self, frame, ref, hw, fw, contacts, midpoints, pinch_candidates, options):
        self.frame, self.ref, self.hw, self.fw = frame, ref, hw, fw
        self.contacts, self.midpoints = contacts, midpoints
        self.pinch_candidates, self.options = pinch_candidates, options

    def raw_offsets(self, uv):
        wref = self.ref(uv)
        return wref+self.hw(uv), wref+self.fw(uv)

    def surfaces(self, uv):
        """Return classified, thickness-controlled HW/FW in local frame."""
        h, f = self.raw_offsets(uv)
        gap = h-f
        # The outside reversed intervals deliberately make gap negative.
        # Negative gap is *not* part of the solid, even though both walls exist.
        active_gap = np.maximum(gap, 0)
        if self.options.min_thickness is not None:
            active_gap = np.maximum(active_gap, self.options.min_thickness)
        if self.options.max_thickness is not None:
            active_gap = np.minimum(active_gap, self.options.max_thickness)
        center = .5*(h+f)
        return center+active_gap/2, center-active_gap/2, gap

    def contains_local(self, uvw):
        xyz = np.asarray(uvw, float)
        h, f, gap = self.surfaces(xyz[..., :2])
        w = xyz[..., 2]
        return (gap > 0) & (w >= f) & (w <= h)

    def contains(self, points_xyz):
        return self.contains_local(self.frame.to_local(points_xyz))

    def surface_meshes(self, umin, umax, vmin, vmax, nu=80, nv=80):
        uv = np.stack(np.meshgrid(np.linspace(umin, umax, nu),
                                  np.linspace(vmin, vmax, nv), indexing="ij"),axis=-1)
        h, f, signed = self.surfaces(uv)
        top = self.frame.to_global(np.concatenate((uv, h[...,None]),axis=-1))
        bottom = self.frame.to_global(np.concatenate((uv, f[...,None]),axis=-1))
        return top, bottom, signed

    def closed_volume_mesh(self, umin, umax, vmin, vmax, *, nu=65, nv=65, nw=35):
        """Marching-cubes solid, including pinchout and box-side caps."""
        from skimage.measure import marching_cubes
        import trimesh
        u = np.linspace(umin, umax, nu)
        v = np.linspace(vmin, vmax, nv)
        uv = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)
        h, f, _ = self.surfaces(uv)
        lo = float(np.min(f)-2)
        hi = float(np.max(h)+2)
        w = np.linspace(lo, hi, nw)
        # Signed min of four side caps, upper/lower walls; positive = inside.
        cap = np.minimum.reduce([
            np.broadcast_to((u[:,None]-umin)[...,None], (nu,nv,nw)),
            np.broadcast_to((umax-u[:,None])[...,None], (nu,nv,nw)),
            np.broadcast_to((v[None,:]-vmin)[...,None], (nu,nv,nw)),
            np.broadcast_to((vmax-v[None,:])[...,None], (nu,nv,nw)),
        ])
        values = np.minimum.reduce([
            np.broadcast_to((w[None,None,:]-f[...,None]),(nu,nv,nw)),
            np.broadcast_to((h[...,None]-w[None,None,:]),(nu,nv,nw)),
            cap,
        ])
        # A nonnegative signed thickness between the surfaces is required.
        values = np.minimum(values, (h-f)[...,None]*.5)
        if not (np.any(values > 0) and np.any(values < 0)):
            raise ValueError("No positive volumetric vein in supplied box")
        verts, faces, _, _ = marching_cubes(
            values.astype(np.float32), level=0,
            spacing=(u[1]-u[0],v[1]-v[0],w[1]-w[0]))
        verts += [umin,vmin,lo]
        xyz = self.frame.to_global(verts)
        mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=True)
        return mesh


def _observations(holes, frame, opts):
    contacts, midpoints = [], []
    thicknesses = []
    for hole in holes:
        intervals = [i for i in hole.logs if i.code == opts.vein_code]
        # Join contiguous vein intervals before contact extraction.
        chains = []
        for seg in intervals:
            if chains and seg.start <= chains[-1][1]+1e-8:
                chains[-1][1] = max(chains[-1][1], seg.end)
            else:
                chains.append([seg.start, seg.end])
        if not opts.multiple_intersections and len(chains) > 1:
            # Explicit approximation, not a proven Leapfrog selection rule.
            chains = [max(chains, key=lambda x: x[1]-x[0])]
        for start, end in chains:
            start_xyz, end_xyz = hole.at(start), hole.at(end)
            midpoint_xyz = hole.at(.5*(start+end))
            # Classify wall polarity by reference-normal projection.
            ordered = sorted([(start, start_xyz), (end, end_xyz)],
                             key=lambda p: float(frame.to_local(p[1])[2]),
                             reverse=True)
            for wall, (md, xyz) in zip(("hw", "fw"), ordered):
                is_end = (abs(md-hole.path[0,0])<1e-8 or abs(md-hole.path[-1,0])<1e-8)
                if is_end and not opts.include_end_wall_points:
                    continue
                contacts.append(Contact(hole.name, md, xyz, wall,
                                        "end" if is_end else "observed"))
            end_segment = abs(start-hole.path[0,0])<1e-8 or abs(end-hole.path[-1,0])<1e-8
            midpoints.append(Midpoint(hole.name, .5*(start+end), midpoint_xyz,
                                      opts.include_end_midpoints or not end_segment))
            thicknesses.append(abs(float(frame.to_local(start_xyz)[2] -
                                         frame.to_local(end_xyz)[2])))
    return contacts, midpoints, thicknesses


def _fit_reference(frame, midpoints, opts):
    if opts.reference == "planar" or sum(m.included for m in midpoints) < 3:
        return lambda uv: np.zeros(np.asarray(uv).shape[:-1])
    xyz = np.array([frame.to_local(m.xyz) for m in midpoints if m.included])
    interp = Interpolant(xyz[:,:2], xyz[:,2], opts.smoothing)
    return interp


def _fit_walls(frame, reference, contacts, smoothing):
    fields = {}
    for wall in ("hw", "fw"):
        values = np.array([frame.to_local(c.xyz) for c in contacts if c.wall == wall])
        if not len(values):
            raise ValueError(f"No contact points found for {wall}")
        fields[wall] = Interpolant(values[:,:2],
                                   values[:,2]-reference(values[:,:2]), smoothing)
    return fields["hw"], fields["fw"]


def _hole_wall_intersection(hole, model, samples=100):
    md = np.linspace(hole.path[0,0], hole.path[-1,0], samples)
    xyz = np.array([hole.at(m) for m in md])
    q = model.frame.to_local(xyz)
    h, f, _ = model.surfaces(q[:,:2])
    candidates = []
    for wall in (h,f):
        val = q[:,2]-wall
        crossing = np.where(val[:-1]*val[1:] <= 0)[0]
        for i in crossing:
            t = abs(val[i]) / max(abs(val[i])+abs(val[i+1]), 1e-12)
            candidates.append(md[i]+t*(md[i+1]-md[i]))
    if not candidates:
        return None
    return float(np.median(candidates))


def _gap_to_logged(hole, md):
    if not hole.logs: return float("inf")
    return float(min(max(i.start-md, 0., md-i.end) for i in hole.logs))


def build_vein(holes: list[Hole], frame: Frame, opts: Options) -> VeinModel:
    contacts, midpoints, thicknesses = _observations(holes, frame, opts)
    reference = _fit_reference(frame, midpoints, opts)
    hfun, ffun = _fit_walls(frame, reference, contacts, opts.smoothing)
    provisional = VeinModel(frame, reference, hfun, ffun, contacts,
                             midpoints, [], opts)
    decisions = []
    if not opts.pinch_out:
        return provisional
    typical = float(np.median([x for x in thicknesses if x>1e-8])) if any(x>1e-8 for x in thicknesses) else 1.
    width = (opts.reversal_halfwidth if opts.reversal_halfwidth is not None
             else max(.15*typical,.01))
    for hole in holes:
        if any(item.code == opts.vein_code for item in hole.logs):
            continue
        md = _hole_wall_intersection(hole, provisional)
        gap = _gap_to_logged(hole, md) if md is not None else float("inf")
        decision = "eligible"
        if hole.name in opts.excluded_pinch_holes:
            decision = "manual-exclusion"
        elif not hole.logs and not opts.pinch_empty_traces:
            decision = "completely-empty-disabled"
        elif md is None:
            decision = "no-wall-intersection"
        elif hole.logs and opts.dont_pinch_far_logged and gap >= opts.logged_distance:
            decision = "logged-interval-too-far"
        decisions.append(PinchCandidate(hole.name, decision, md, gap))
        if decision != "eligible":
            continue
        xyz = hole.at(md)
        q = frame.to_local(xyz)
        uv = q[:2][None,:]
        # A provisional centre anchors the synthetic reversed wall constraints.
        hc, fc = provisional.raw_offsets(uv)
        centre = float((hc[0]+fc[0])/2.)
        for wall, shifted in (("hw", -width), ("fw", width)):
            new_xyz = frame.to_global([q[0], q[1], centre+shifted])
            contacts.append(Contact(hole.name, md, new_xyz, wall, "reversed-outside"))
    hfun, ffun = _fit_walls(frame, reference, contacts, opts.smoothing)
    return VeinModel(frame, reference, hfun, ffun, contacts,
                     midpoints, decisions, opts)
