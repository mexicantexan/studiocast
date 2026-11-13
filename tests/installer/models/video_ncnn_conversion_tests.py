#!/usr/bin/env python3
"""Hermetic converter security/schema checks; real numerical matrices are offline opt-in."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("video_ncnn",ROOT/"scripts/models/convert_video_ncnn.py")
converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)

class ConversionTests(unittest.TestCase):
    SIMPLE="7767517\n3 3\nInput a 0 1 in0\nInput b 0 1 in1\nBinaryOp sum 2 1 in0 in1 out0 0=0\n"
    def test_exact_native_names(self):
        text=converter.normalize_param(self.SIMPLE,"fastdvdnet")
        self.assertIn("noisy noise_map denoised",text)
        self.assertNotIn(" in0",text)
    def test_false_success_operator_rejected(self):
        for kind in ("Gather","Tensor.to","PythonOp","MultiHeadAttention"):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                converter.normalize_param(self.SIMPLE.replace("BinaryOp",kind),"fastdvdnet")
    def test_topological_and_signature_closure(self):
        for changed in (self.SIMPLE.replace("in0 in1 out0","missing in1 out0"),
                        self.SIMPLE.replace("in0 in1 out0","in0 in1 in0"),
                        self.SIMPLE.replace("3 3","4 3"),
                        self.SIMPLE.replace("in1","in2")):
            with self.assertRaises(ValueError):converter.normalize_param(changed,"fastdvdnet")
    def test_dynamic_quantized_weight_rejection(self):
        for suffix in ("8=1","19=1"):
            changed=self.SIMPLE.replace("BinaryOp sum","Convolution sum").replace("0=0",suffix)
            with self.assertRaises(ValueError):converter.normalize_param(changed,"fastdvdnet")
    def test_flx_rewrites_are_exact_bounded_pair(self):
        text="7767517\n7 8\nInput a 0 1 in0\nInput b 0 1 in1\nInput c 0 1 in2\ntorch.tile d 1 1 in2 angle\nAveragePool e 1 1 angle pooled\nConcat f 3 1 in0 in1 pooled joined\nSlice g 1 2 joined out0 out1 1=0\n"
        normalized=converter.normalize_param(text,"left")
        self.assertIn("Tile d 1 1 angles:0 angle 0=0 1=3072",normalized)
        self.assertIn("Pooling e",normalized)
        self.assertIn("flow lcm",normalized)
        with self.assertRaises(ValueError):converter.normalize_param(text.replace("AveragePool","Pooling"),"left")
    def test_input_hash_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"source";p.write_bytes(b"verified")
            converter.verify(p,converter.sha256(p))
            link=Path(tmp)/"link";link.symlink_to(p)
            with self.assertRaises(ValueError):converter.verify(link,converter.sha256(p))
            with self.assertRaises(ValueError):converter.verify(p,"0"*64)
    def test_native_rank_and_dynamic_geometry_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"artifact";p.write_bytes(b"test")
            fast=converter.network_spec("fastdvdnet",p,p,p)
            self.assertEqual(fast["inputs"][0],dict(blob="noisy",dims=3,w=0,h=0,c=15,dynamic_hw=True))
            eye=converter.network_spec("left",p,p,p)
            self.assertEqual(eye["inputs"][2],dict(blob="angles:0",dims=1,w=2,h=1,c=1))
            self.assertEqual([x["blob"] for x in eye["outputs"]],["flow","lcm"])
            yunet=converter.network_spec("yunet",p,p,p)
            self.assertEqual(len(yunet["outputs"]),12)
            self.assertEqual(yunet["outputs"][6],dict(blob="bbox_8",dims=2,w=4,h=6400,c=1))
    def test_exact_sampler_border_and_numerical_fail_closed(self):
        import numpy as np
        image=np.ones((1,3,48,64),np.float32)
        flow=np.zeros((1,2,48,64),np.float32)
        lcm=np.zeros((1,2,48,64),np.float32);lcm[:,0]=1
        # Source's clipping-before-weight rule cancels at the right/bottom edge.
        grid=np.ones((2,3072),np.float32)
        actual=converter.flx_rgb([image],[flow,lcm],grid)
        np.testing.assert_array_equal(actual,np.zeros_like(image))
        with self.assertRaises(ValueError):converter.compare(np.full((1,),np.nan),np.ones((1,)),"nan")
        with self.assertRaises(ValueError):converter.compare(np.zeros((1,)),np.ones((1,)),"mismatch")

if __name__=="__main__":unittest.main()
