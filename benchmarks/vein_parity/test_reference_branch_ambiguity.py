"""Synthetic zero-private-data tests for folded-reference sheet crossings."""
import unittest
import numpy as np
from reference_branch_ambiguity import vertical_reference_branches,pick_nearest_reference_branch


class TestReferenceBranchAmbiguity(unittest.TestCase):
    def test_common_diagonal_deduplicated(self):
        v=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        b=vertical_reference_branches(v,[[0,1,2],[0,2,3]],0,0)
        self.assertEqual(len(b),1)
        self.assertEqual(len(b[0].face_ids),2)
        self.assertFalse(b[0].ambiguous_polarity)
        self.assertEqual(b[0].normal_z_signs,(1,))

    def test_stacked_overturned_sheets_are_distinct(self):
        v=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                    [-1,-1,20],[1,-1,20],[1,1,20],[-1,1,20]],float)
        f=[[0,1,2],[0,2,3],[4,6,5],[4,7,6]]
        b=vertical_reference_branches(v,f,0.12,0.16)
        self.assertEqual(len(b),2)
        self.assertEqual([s.normal_z_signs for s in b],[(1,),(-1,)])
        self.assertEqual(pick_nearest_reference_branch(b,18).z,20.)

    def test_same_depth_opposite_orientations_flagged(self):
        v=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        b=vertical_reference_branches(v,[[0,1,2],[0,2,1]],0.1,-0.4)
        self.assertEqual(len(b),1)
        self.assertTrue(b[0].ambiguous_polarity)
        self.assertEqual(b[0].normal_z_signs,(-1,1))

    def test_missed_trace_requires_separate_fallback(self):
        v=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],float)
        b=vertical_reference_branches(v,[[0,1,2],[0,2,3]],5,5)
        self.assertEqual(b,[])
        self.assertIsNone(pick_nearest_reference_branch(b,0))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            vertical_reference_branches([[0,0],[1,1]],[[0,1,2]],0,0)


if __name__=="__main__": unittest.main()
