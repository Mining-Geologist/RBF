"""Read-only saved native reference-input provenance audit.

Given an OWNER-SUPPLIED Leapfrog .aproj file, statically inspect persistent
object opcode streams without importing vendor classes or executing pickle.

Explicitly checks the reference's midpoint source and boundary object for
names of possible non-positional orientation, trend, selection or clipping
inputs. Metadata field *names* are printed; numerical data and geological
coordinates are NOT exposed, copied, or committed.

Usage:
    python reference_input_provenance_audit.py path/to/Vein_Benchmark.aproj
    python reference_input_provenance_audit.py path/to/Vein_Benchmark.aproj --oid 4554
"""
import argparse
import json
import re
from native_reference_dag_audit import ProjectGraph


KEY_PAT = re.compile(r"^[_a-zA-Z][_a-zA-Z0-9]{1,79}$")
IMPORTANT = re.compile(
    r"orient|normal|azimuth|dip|strike|polarity|trend|plane|bound|"
    r"filter|location|point|mid|input|merge|segment|intersect|"
    r"resolution|ref|invert|geol|rotation|transform|weight|"
    r"selection|exclude|source|curve|mesh", re.I
)


def string_field_candidates(operations):
    """Heuristic *superset* of field names in the serialized object.

    Results may include enum values or member names not used as keys.
    This is for manual code navigation, not for decoding data.
    """
    strings = set()
    for op, value, _ in operations:
        if op.name.endswith("UNICODE") and isinstance(value, str):
            if KEY_PAT.fullmatch(value) and IMPORTANT.search(value):
                strings.add(value)
    return sorted(strings)


def inspect_reference(g, reference_oid, max_hops=2):
    cls = g.classes.get(reference_oid, '')
    if not cls.endswith(".VeinReferenceSurfaceBlock"):
        raise ValueError("Requested ID is not a saved native vein reference block")
    pending = [(reference_oid, 0, "reference")]
    seen = set()
    discovered = []
    while pending:
        oid, hop, role = pending.pop(0)
        if oid in seen or oid not in g.classes:
            continue
        seen.add(oid)
        direct = g.inputs(oid)
        fields = string_field_candidates(g.ops.get(oid, ()))
        # Only treat known decoded OIDs as real dependencies. Generic string
        # matches are candidate fields; do not infer usage or direction.
        item = {
            "oid":oid,
            "class":g.classes[oid],
            "role":role,
            "hop":hop,
            "declared_inputs":[{"oid":i,"class":g.classes.get(i, '')}
                               for i in direct],
            "candidate_orientation_or_selection_field_names":fields,
        }
        if role=="reference":
            item["known_attributes"]={
                key:g.attr(oid,key)
                for key in ("points","hw_points","fw_points","raw_vein",
                            "custom_mesh","face_dip_values","dip_azimuth_values")
            }
        discovered.append(item)
        if hop < max_hops:
            for i in direct:
                pending.append((i,hop+1,"declared input of "+str(oid)))
            # The documented direct point source is not always serialized
            # in _inputs; inspecting it explicitly is non-causal and flagged.
            if role=="reference":
                for name in ("points","raw_vein"):
                    v=g.attr(oid,name)
                    if v in g.classes:
                        pending.append((v,hop+1,"attribute '"+name+"' of "+str(oid)))
    return {"reference_oid":reference_oid,"visited_objects":discovered,
            "warning":"Field-name matches do not prove algorithmic usage. "
                      "A raw_vein attribute is not necessarily upstream."}


def audit(project, reference_oid=None):
    g=ProjectGraph(project)
    refs=g.find(".VeinReferenceSurfaceBlock")
    if reference_oid is not None:
        refs=[reference_oid]
    return {"total_saved_vein_reference_objects":len(g.find(".VeinReferenceSurfaceBlock")),
            "models":[inspect_reference(g,oid) for oid in refs]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project")
    p.add_argument("--oid",type=int)
    p.add_argument("--output",help="Local JSON path; don't commit proprietary project results")
    args=p.parse_args()
    result=json.dumps(audit(args.project,args.oid),indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(result,encoding="utf8")
    print(result)
