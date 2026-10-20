"""Unit tests using synthetic pickle-opcode streams, no owner project files."""
import pickle,pickletools,unittest
from reference_input_provenance_audit import (
    string_field_candidates, inspect_reference,KEY_PAT, IMPORTANT)


def ops(payload):
    return list(pickletools.genops(pickle.dumps(payload,protocol=4)))


class FakeGraph:
    def __init__(self):
        self.classes={
            4554:"engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock",
            4550:"engine.locations.MergedLocationsBlock",
            4336:"engine.boundary.FaultBlockBoundaryBlock",
        }
        self.ops={
            4554:ops({"points":"placeholder","surface_type":"curved",
                      "resolution":20,"_inputs":"map"}),
            4550:ops({"selected_inputs":"active","exclude_midpoints":False}),
            4336:ops({"plane_normal":[0,0,1],"boundary_filter":True})
        }
    def inputs(self,oid):return {4554:[4550,4336]}.get(oid,[])
    def field_oid(self,oid,name):
        return {("4554","points"):4550}.get((str(oid),name))


class ProvenanceTests(unittest.TestCase):
    def test_field_names_reveal_candidate_fields_not_values(self):
        names=string_field_candidates(ops({
            "boundary_filter":True,"plane_normal":[0,0,1],
            "foo":2,"selected_inputs":1,"x":[6],"geology_query":"Excluded",
        }))
        self.assertIn("boundary_filter",names)
        self.assertIn("plane_normal",names)
        self.assertIn("selected_inputs",names)
        self.assertNotIn("foo",names)
    def test_hops_include_declared_boundary_and_point_source(self):
        record=inspect_reference(FakeGraph(),4554)
        classes=[item["class"].split(".")[-1] for item in record["visited_objects"]]
        self.assertIn("MergedLocationsBlock",classes)
        self.assertIn("FaultBlockBoundaryBlock",classes)
        boundary=next(x for x in record["visited_objects"] if x["oid"]==4336)
        self.assertIn("plane_normal",boundary["candidate_orientation_or_selection_field_names"])
    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError): inspect_reference(FakeGraph(),4336)


if __name__=="__main__":
    unittest.main()
