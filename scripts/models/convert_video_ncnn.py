#!/usr/bin/env python3
"""Offline, pinned FastDVDnet/YuNet/FLX conversion; no install or publication.

Requires existing local sources, pnnx, full license inputs and the CPU reference
built from video_ncnn_reference.cpp against the lock's exact clean ncnn source.
The original ONNX artifacts/IDs/defaults remain unchanged. --output must be new.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(Path(__file__).with_name("video-ncnn.lock.json").read_text())
VULKAN_TYPES = frozenset(("Input", "Split", "Convolution", "ConvolutionDepthWise",
    "BinaryOp", "BatchNorm", "Slice", "Concat", "ReLU", "Interp", "InnerProduct",
    "Pooling", "Reshape", "Permute", "Sigmoid", "PixelShuffle", "Tile", "TanH", "Softmax"))
PACKS = {"fastdvdnet": "video_denoise/Good Quality (FastDVDnet sigma15)",
         "yunet": "face_detection/Best Quality (YuNet FP32)",
         "left": "eye_contact/Better Quality (Gaze Correction Cam v0.1.1 FP32)",
         "right": "eye_contact/Better Quality (Gaze Correction Cam v0.1.1 FP32)"}
INPUTS = {"fastdvdnet": ["noisy", "noise_map"], "yunet": ["input"],
          "left": ["eye_rgb:0", "anchors:0", "angles:0"],
          "right": ["eye_rgb:0", "anchors:0", "angles:0"]}
OUTPUTS = {"fastdvdnet": ["denoised"],
           "yunet": [f"{kind}_{stride}" for kind in ("cls", "obj", "bbox", "kps") for stride in (8, 16, 32)],
           "left": ["flow", "lcm"], "right": ["flow", "lcm"]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, digest):
    if path.is_symlink() or not path.is_file() or sha256(path) != digest:
        raise ValueError(f"input integrity failure: {path}")


def normalize_param(text, name):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != "7767517":
        raise ValueError("invalid ncnn header")
    count, blobs = map(int, lines[1].split())
    produced, consumed, rows, inputs = set(), set(), [], []
    rewrites = {"torch.tile": 0, "AveragePool": 0}
    aliases = {f"in{i}": v for i, v in enumerate(INPUTS[name])}
    aliases.update({f"out{i}": v for i, v in enumerate(OUTPUTS[name])})
    for line in lines[2:]:
        t = line.split()
        if len(t) < 4:
            raise ValueError("invalid ncnn layer")
        n_in, n_out = map(int, t[2:4]); end = 4 + n_in + n_out
        if min(n_in, n_out) < 0 or len(t) < end:
            raise ValueError("invalid ncnn layer arity")
        if t[0] in rewrites:
            if name not in ("left", "right") or n_in != 1 or n_out != 1 or len(t) != end:
                raise ValueError("unreviewed conversion rewrite")
            kind = t[0]; rewrites[kind] += 1
            if kind == "torch.tile":
                t[0] = "Tile"; t.extend(("0=0", "1=3072"))
            else:
                t[0] = "Pooling"
                t.extend(("0=1", "1=2", "11=2", "2=1", "12=1", "3=0",
                          "14=1", "13=0", "15=1", "4=0", "5=1", "6=0"))
        if t[0] not in VULKAN_TYPES:
            raise ValueError(f"unreviewed/non-Vulkan ncnn operator: {t[0]}")
        bottom, top = t[4:4+n_in], t[4+n_in:end]
        if any(x not in produced for x in bottom) or len(set(top)) != len(top) or any(x in produced for x in top):
            raise ValueError("ncnn graph is not topologically complete")
        if t[0] == "Input":
            if n_in or n_out != 1:
                raise ValueError("invalid ncnn input")
            inputs.extend(top)
        if t[0] in ("Convolution", "ConvolutionDepthWise", "InnerProduct"):
            params = dict(x.split("=", 1) for x in t[end:])
            if params.get("8", "0") != "0" or params.get("19", "0") != "0":
                raise ValueError("quantized/dynamic weights forbidden")
        produced.update(top); consumed.update(bottom)
        t[4:end] = [aliases.get(x, x) for x in t[4:end]]
        rows.append(" ".join(t))
    if inputs != [f"in{i}" for i in range(len(INPUTS[name]))] or produced-consumed != {f"out{i}" for i in range(len(OUTPUTS[name]))}:
        raise ValueError("incorrect ncnn input/output signature")
    if len(rows) != count or len(produced) != blobs:
        raise ValueError("incorrect ncnn graph counts")
    if name in ("left", "right") and rewrites != {"torch.tile": 1, "AveragePool": 1}:
        raise ValueError("FLX prefix rewrite count changed")
    return "\n".join((lines[0], lines[1], *rows)) + "\n"


def load_onnx(path, name):
    import onnx
    m = onnx.load(path, load_external_data=False)
    if any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer):
        raise ValueError("external ONNX data forbidden")
    onnx.checker.check_model(m, full_check=True)
    if [x.name for x in m.graph.input] != INPUTS[name] or any(x.type.tensor_type.elem_type != 1 for x in [*m.graph.input, *m.graph.output]):
        raise ValueError("source tensor name/type mismatch")
    return m


def meshgrid(model):
    import numpy as np
    import onnx
    grid = next(onnx.numpy_helper.to_array(x) for x in model.graph.initializer
                if x.name == "warping_model/apply_transformation/meshgrid/concat:0")
    if grid.shape != (2, 3072) or hashlib.sha256(grid.astype("<f4").tobytes()).hexdigest() != LOCK["flx_meshgrid_sha256"]:
        raise ValueError("FLX meshgrid identity mismatch")
    grid = grid.reshape(2, 48, 64)
    x, y = grid[0, 0].copy(), grid[1, :, 0].copy()
    rebuilt = np.stack((np.broadcast_to(x, (48, 64)), np.broadcast_to(y[:, None], (48, 64))))
    if rebuilt.tobytes() != grid.tobytes():
        raise ValueError("meshgrid cannot be reconstructed from exact axis bits")
    return grid.reshape(2, -1), dict(x_u32=x.view(np.uint32).tolist(), y_u32=y.view(np.uint32).tolist())


def extract_prefix(source, destination, model):
    import onnx
    from onnx import helper, TensorProto
    meshgrid(model)  # Exact source initializer retained by native shader constants.
    model.graph.node.extend((helper.make_node("Transpose", ["warping_model/Tanh:0"], ["flow"], perm=[0,3,1,2]),
                             helper.make_node("Identity", ["warping_model/lcm_module/Softmax:0"], ["lcm"])))
    del model.graph.output[:]
    model.graph.output.extend(helper.make_tensor_value_info(x, TensorProto.FLOAT, [1,2,48,64]) for x in ("flow", "lcm"))
    unpruned = destination.with_name("unpruned.onnx")
    onnx.save(model, unpruned)
    onnx.utils.extract_model(str(unpruned), str(destination), INPUTS["left"], OUTPUTS["left"])
    unpruned.unlink()


def flx_rgb(inputs, outputs, grid):
    """Literal source ONNX sampler/light math, including clipped-neighbor weights."""
    import numpy as np
    flow, lcm = outputs
    xy = flow.reshape(2, -1) + grid
    px, py = (xy[0]+np.float32(1))*np.float32(32), (xy[1]+np.float32(1))*np.float32(24)
    x, y = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
    x0, x1, y0, y1 = np.clip(x,0,63), np.clip(x+1,0,63), np.clip(y,0,47), np.clip(y+1,0,47)
    xf0, xf1, yf0, yf1 = (v.astype(np.float32) for v in (x0,x1,y0,y1))
    img = inputs[0].reshape(3, -1)
    # Preserve AddN ordering from the ONNX export.
    result = ((xf1-px)*(yf1-py))[None,:]*img[:,y0*64+x0]
    result += ((xf1-px)*(py-yf0))[None,:]*img[:,y1*64+x0]
    result += ((px-xf0)*(yf1-py))[None,:]*img[:,y0*64+x1]
    result += ((px-xf0)*(py-yf0))[None,:]*img[:,y1*64+x1]
    return result.reshape(1,3,48,64)*lcm[:,0:1]+lcm[:,1:2]


def compare(actual, expected, name, max_error=2e-4, mae_limit=2e-5):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"nonfinite/incorrect tensor: {name}")
    diff = np.abs(actual-expected); maximum, mae = float(diff.max()), float(diff.mean())
    if maximum > max_error or mae > mae_limit:
        raise ValueError(f"numerical mismatch {name}: max={maximum}, mae={mae}")
    return dict(case=name, shape=list(actual.shape), max_abs_error=maximum, mean_abs_error=mae)


def reference_run(reference, param, binary, inputs, names_in, names_out, expected, work, case):
    import numpy as np
    cmd = [str(reference), str(param), str(binary), str(len(inputs))]
    for i, a in enumerate(inputs):
        path = work / f"{case}.input{i}.f32"; a.astype("<f4").tofile(path)
        w, h, c = (a.shape[3],a.shape[2],a.shape[1]) if a.ndim == 4 else (a.shape[-1],1,1)
        cmd.extend((names_in[i],str(w),str(h),str(c),str(path)))
    cmd.append(str(len(names_out)))
    paths = []
    for i, name in enumerate(names_out):
        p = work / f"{case}.actual{i}.f32"; paths.append(p); cmd.extend((name,str(p)))
    subprocess.run(cmd, check=True, timeout=180, stdout=subprocess.DEVNULL)
    actual = []
    for i, (path, gold) in enumerate(zip(paths, expected)):
        gold.astype("<f4").tofile(work / f"{case}.expected{i}.f32")
        actual.append(np.fromfile(path, dtype="<f4").reshape(gold.shape))
    return actual


def rgb_cases(h, w, scale=1):
    import numpy as np
    y,x = np.indices((h,w), dtype=np.float32)
    rng = np.random.default_rng(20261112)
    return {"black":np.zeros((1,3,h,w),np.float32),
            "white":np.full((1,3,h,w),scale,np.float32),
            "gradient":np.stack((x/max(w-1,1),y/max(h-1,1),(x+y)/max(w+h-2,1)))[None]*scale,
            "checker":np.broadcast_to(((x.astype(int)//4+y.astype(int)//4)%2)[None,None],(1,3,h,w)).astype(np.float32)*scale,
            "seeded_noise":rng.random((1,3,h,w),dtype=np.float32)*scale}


def numerical(name, source, converted_onnx, param, binary, reference, work):
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=1
    original=ort.InferenceSession(str(source),options,providers=["CPUExecutionProvider"])
    prefix=ort.InferenceSession(str(converted_onnx),options,providers=["CPUExecutionProvider"]) if name in ("left","right") else original
    rows=[]; fixtures=[]
    cases=[]
    if name == "fastdvdnet":
        # Same param/bin at all geometries; odd source extents edge-pad on GPU.
        for h,w in ((8,12),(48,64),(51,67)):
            ph,pw=(h+3)//4*4,(w+3)//4*4
            for label,rgb in rgb_cases(h,w).items():
                image=np.pad(rgb,((0,0),(0,0),(0,ph-h),(0,pw-w)),mode="edge")
                noisy=np.concatenate([image*.85,image*.95,image,image,image],axis=1)
                for sigma in (0,15,55):
                    cases.append((f"{label}-{h}x{w}-s{sigma}",[noisy,np.full((1,1,ph,pw),sigma/255,np.float32)]))
    elif name == "yunet":
        cases=[(label,[image]) for label,image in rgb_cases(640,640,255).items()]
    else:
        y,x=np.indices((48,64),dtype=np.float32)
        points=[(11,24),(21,16),(42,16),(53,24),(42,31),(21,31)]
        if name == "left": points=[points[i] for i in (3,2,1,0,5,4)]
        anchors=np.stack([axis-v for px,py in points for axis,v in ((x,px),(y,py))])[None]
        for label,image in rgb_cases(48,64).items():
            for angle in (0,-.05,-.1):
                cases.append((f"{label}-angle{angle}",[image,anchors,np.array([[0,angle]],np.float32)]))
    grid=meshgrid(load_onnx(source,name))[0] if name in ("left","right") else None
    for case,inputs in cases:
        feed=dict(zip(INPUTS[name],inputs)); golden=prefix.run(None,feed)
        actual=reference_run(reference,param,binary,inputs,INPUTS[name],OUTPUTS[name],golden,work,case)
        if grid is not None:
            rgb=original.run(None,feed)[0]
            # No relaxed oracle: exact source decomposition must agree before
            # comparing the converted model's completed RGB result.
            rows.append(compare(flx_rgb(inputs,golden,grid),rgb,case+"-exact-tail",0,0))
            rows.append(compare(flx_rgb(inputs,actual,grid),rgb,case+"-rgb"))
            rgb.astype("<f4").tofile(work/f"{case}.rgb.expected.f32")
            flx_rgb(inputs,actual,grid).astype("<f4").tofile(work/f"{case}.rgb.actual.f32")
        else:
            for i,(a,b) in enumerate(zip(actual,golden)): rows.append(compare(a,b,case+"-"+OUTPUTS[name][i]))
        fixtures.append(dict(case=case,input_shapes=[list(a.shape) for a in inputs],output_shapes=[list(a.shape) for a in golden]))
    (work/"fixtures.json").write_text(json.dumps(dict(inputs=INPUTS[name],outputs=OUTPUTS[name],cases=fixtures),indent=2)+"\n")
    return rows


def tensor(blob,dims,w,h=1,c=1,dynamic=False):
    t=dict(blob=blob,dims=dims,w=w,h=h,c=c)
    if dynamic:t["dynamic_hw"]=True
    return t


def network_spec(name,param,binary,source):
    if name == "fastdvdnet":
        ins=[tensor("noisy",3,0,0,15,True),tensor("noise_map",3,0,0,1,True)];outs=[tensor("denoised",3,0,0,3,True)];profile="fastdvdnet_rgb_v1"
    elif name == "yunet":
        ins=[tensor("input",3,640,640,3)];outs=[tensor(f"{k}_{s}",2,c,(640//s)**2) for k,c in (("cls",1),("obj",1),("bbox",4),("kps",10)) for s in (8,16,32)];profile="yunet_640_v1"
    else:
        ins=[tensor("eye_rgb:0",3,64,48,3),tensor("anchors:0",3,64,48,12),tensor("angles:0",1,2)];outs=[tensor(n,3,64,48,2) for n in ("flow","lcm")];profile="flx_prefix_v1"
    return dict(role=name if name in ("left","right") else "main",profile=profile,
                param_file=param.name,bin_file=binary.name,source_onnx_sha256=sha256(source),
                converter=dict(name="pnnx",version=LOCK["pnnx_version"]),precision="fp32",inputs=ins,outputs=outs)


def convert(args):
    sources={n:getattr(args,n).absolute() for n in PACKS}
    for n,p in sources.items():verify(p,LOCK["sources"][n])
    verify(args.pnnx,LOCK["pnnx_sha256"])
    for n in ("yunet","flx"):verify(getattr(args,n+"_license"),LOCK["licenses"][n]["sha256"])
    import numpy,onnx,onnxruntime
    versions={"numpy":numpy.__version__,"onnx":onnx.__version__,"onnxruntime":onnxruntime.__version__}
    if versions != LOCK["python_versions"]:raise ValueError("use locked Python dependencies")
    desc=json.loads(subprocess.check_output([str(args.reference),"--describe"],text=True,timeout=10))
    if desc.get("protocol") != "studiocast-video-ncnn-reference-v1" or not desc.get("avx2") or not desc.get("fma") or desc.get("runtime_cpu"):
        raise ValueError("reference must use locked AVX2/FMA direct-convolution build")
    args.output.mkdir(parents=True,exist_ok=False)
    reports={};manifests={};catalog=json.loads((ROOT/"installer/models/curated-model-catalog-v1.json").read_text())
    for name,source in sources.items():
        work=args.output/"work"/name;work.mkdir(parents=True)
        original=load_onnx(source,name);model=work/"model.onnx"
        if name in ("left","right"):extract_prefix(source,model,original)
        else:shutil.copyfile(source,model)
        shapes={"fastdvdnet":"[1,15,48,64],[1,1,48,64]","yunet":"[1,3,640,640]","left":"[1,3,48,64],[1,12,48,64],[1,2]","right":"[1,3,48,64],[1,12,48,64],[1,2]"}
        with (work/"conversion.log").open("w") as log:
            subprocess.run([str(args.pnnx),"model.onnx","inputshape="+shapes[name],"fp16=0","optlevel=2"],cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        param=work/"model.ncnn.param";binary=work/"model.ncnn.bin"
        param.write_text(normalize_param(param.read_text(),name))
        numeric=work/"numerical";numeric.mkdir()
        rows=numerical(name,source,model,param,binary,args.reference,numeric)
        relative="open_video/"+PACKS[name];pack=args.output/"source"/relative;pack.mkdir(parents=True,exist_ok=True)
        template=ROOT/"resources/model_packs"/relative
        if relative not in manifests:
            manifest=json.loads((template/"model.json").read_text())
            manifest["files"]=[f for f in manifest["files"] if f["kind"] not in ("ncnn_param","ncnn_bin")]
            manifest.pop("ncnn_vulkan_networks",None)
            manifests[relative]=manifest
        manifest=manifests[relative]
        onnx_name=next(f["name"] for f in manifest["files"] if f["kind"]=="onnx" and f["role"]==(name if name in ("left","right") else "main"))
        shutil.copyfile(source,pack/onnx_name)
        stem="gaze_flx_"+name if name in ("left","right") else "model"
        dst_param,dst_bin=pack/(stem+".ncnn.param"),pack/(stem+".ncnn.bin")
        shutil.copyfile(param,dst_param);shutil.copyfile(binary,dst_bin)
        role="vulkan_"+(name if name in ("left","right") else "main")
        manifest["files"].extend(dict(name=p.name,kind=k,role=role,sha256=sha256(p)) for p,k in ((dst_param,"ncnn_param"),(dst_bin,"ncnn_bin")))
        manifest.setdefault("ncnn_vulkan_networks",[]).append(network_spec(name,dst_param,dst_bin,source))
        if name == "fastdvdnet":license_text=(template/"LICENSE.txt").read_text().split("\nConversion notice:",1)[0]
        else:
            license_text=(template/"LICENSE.txt").read_text().split("\nConversion notice:",1)[0].split("\nFull upstream license at ",1)[0]
            license_text+="\nFull upstream license at "+LOCK["licenses"]["flx" if name in ("left","right") else "yunet"]["upstream_revision"]+":\n\n"+getattr(args,("flx" if name in ("left","right") else "yunet")+"_license").read_text()
        license_text+="\nConversion notice: Original ONNX retained. FP32 ncnn conversion via pnnx "+LOCK["pnnx_version"]+". FLX learned-prefix flow/lcm and exact native sampler/lighting preserve original corrected-RGB semantics. No retraining.\n"
        (pack/"LICENSE.txt").write_text(license_text)
        reports[name]=dict(equivalence=rows,artifacts={p.name:dict(size_bytes=p.stat().st_size,sha256=sha256(p)) for p in (dst_param,dst_bin)})
    for relative,manifest in manifests.items():
        pack=args.output/"source"/relative;(pack/"model.json").write_text(json.dumps(manifest,indent=2)+"\n")
        entry=next(p for p in catalog["packs"] if "open_video/"+p["pack_path"]==relative)
        original_names={a["name"] for a in entry["artifacts"]}
        # Original source candidates and hashes are preserved for active catalog
        # review; candidate transactions bind every file to their own source tree.
        for artifact in entry["artifacts"]:artifact["source_paths"]=[relative+"/"+artifact["name"]]
        for file in manifest["files"]:
            if file["name"] not in original_names:
                p=pack/file["name"];entry["artifacts"].append(dict(name=p.name,sha256=sha256(p),size_bytes=p.stat().st_size,size_status="known",source_paths=[relative+"/"+p.name]))
        for meta in entry["metadata"]:
            p=pack/meta["destination"];meta["sha256"]=sha256(p);dst=args.output/"metadata"/meta["source"];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    catalog["catalog_version"]="2026-11-12-vulkan-all-candidate"
    (args.output/"curated-model-catalog-v1.candidate.json").write_text(json.dumps(catalog,indent=2)+"\n")
    _,axes=meshgrid(load_onnx(sources["left"],"left"));(args.output/"flx-meshgrid-exact-axis-bits.json").write_text(json.dumps(axes,indent=2)+"\n")
    report=dict(status="candidate_unpublished",lock=LOCK,tools=versions,reference_description=desc,
                reference_sha256=sha256(args.reference),models=reports,gpu_qualification="not_run",
                acceptance=dict(max_abs_error=2e-4,mean_abs_error=2e-5,exact_flx_decomposition=True))
    (args.output/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in (*PACKS,"pnnx","reference","yunet_license","flx_license","output"):
        p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    for n,v in vars(args).items():setattr(args,n,v.absolute())
    print(json.dumps(convert(args),indent=2))

if __name__=="__main__":main()
