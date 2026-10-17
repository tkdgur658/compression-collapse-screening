"""Generates the numeric LaTeX tables of the appendix directly from the SUMMARY.md files (no hand-typed numbers).
Run: python -s gen_appendix_tables.py  -> writes ../text/appendix_generated.tex and prints a check log."""
import os as _ccs_os  # path configuration of the released copy (README, Configuration)
CCS_REPO_ROOT = _ccs_os.path.normpath(_ccs_os.path.join(_ccs_os.path.dirname(_ccs_os.path.abspath(__file__)), '../..'))
CCS_RESULTS_ROOT = _ccs_os.environ.get('CCS_RESULTS', _ccs_os.path.join(CCS_REPO_ROOT, 'results'))
import re, os

LOOP = _ccs_os.path.join(CCS_RESULTS_ROOT, 'loop', 'done') + '/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "text", "appendix_generated.tex")
DS = ["PH2", "isic17", "isic18", "kvasir", "cvc", "etis"]
NAME = {"PH2": "PH2", "isic17": "ISIC 2017", "isic18": "ISIC 2018", "kvasir": "Kvasir-SEG",
        "cvc": "CVC-ClinicDB", "etis": "ETIS-Larib"}
BASE = ["maresunet", "utanet", "mk_unet", "mhorunet", "tinyunet", "ucmnet", "lbunet", "cmunext", "egeunet", "malunet"]
MN = {"pbe": "PBE-Net", "maresunet": "MAResUNet", "utanet": "UTANet", "mk_unet": "MK-UNet", "mhorunet": "MHorUNet",
      "tinyunet": "TinyU-Net", "ucmnet": "UCMNet", "lbunet": "LB-UNet", "cmunext": "CMUNeXt", "egeunet": "EGE-UNet",
      "malunet": "MALUNet"}


def read(p):
    return open(p, encoding="utf8").read()


def tables(text):
    """{section: (header, {model: {col: float}})} for '## name' sections that contain a mean-IoU table"""
    out = {}
    for sec in re.split(r"\n## ", "\n" + text)[1:]:
        name = sec.split(" ")[0].strip()
        lines = sec.split("\n")
        for i, l in enumerate(lines):
            if l.startswith("| model |"):
                hdr = [c.strip() for c in l.strip("|").split("|")][1:]
                rows = {}
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    c = [x.strip() for x in lines[j].strip("|").split("|")]
                    vals = {}
                    for h, v in zip(hdr, c[1:]):
                        m = re.match(r"([\d.]+)", v)
                        vals[h] = float(m.group(1)) if m else None
                    rows[c[0]] = vals
                    j += 1
                out[name] = (hdr, rows)
                break
    return out


# ---- compression (9101 weights, 9601/9602 pruning)
w = read(LOOP + "9101_gpu_precision_weight_rtn/SUMMARY.md")
tw = tables(w)
tp = {**tables(read(LOOP + "9601_gpu_prune_skin/SUMMARY.md")), **tables(read(LOOP + "9602_gpu_prune_polyp/SUMMARY.md"))}
rowsA = []
for d in DS:
    rw = tw[d][1]
    rp = tp[d][1]

    def cell(rows, col):
        p = rows["pbe"][col]
        return p, sum(1 for b in BASE if rows[b][col] > p)

    rowsA.append((d, rw["pbe"]["orig"], cell(rw, "w4_pt"), cell(rw, "w3_pt"), cell(rp, "l50"), cell(rw, "w4_pc")))


def tex_compress():
    s = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{\label{tab:app_compress}Exploratory post-training compression of PBE-Net on the validation split of split~1 (single checkpoint per dataset, no calibration or fine-tuning). Each cell gives the mean per-image validation IoU of PBE-Net followed, in parentheses, by the number of the ten baselines that obtain a higher IoU than PBE-Net under the same compression. W4 and W3: symmetric round-to-nearest weight quantization to 4 and 3 bits with one scale per tensor; W4-ch: the same at 4 bits with one scale per output channel; L50: layer-wise unstructured magnitude pruning at 50\% sparsity. Values are copied by script from the machine-written summaries of the exploratory packages 9101, 9601 and 9602.}",
         r"\begin{tabular}{lccccc}", r"\hline",
         r"Dataset & Uncompressed & W4 & W3 & L50 & W4-ch \\", r"\hline"]
    for d, o, c4, c3, cl, c4c in rowsA:
        f = lambda c: "%.3f (%d)" % (c[0], c[1])
        s.append("%s & %.3f & %s & %s & %s & %s \\\\" % (NAME[d], o, f(c4), f(c3), f(cl), f(c4c)))
    s += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(s)


# ---- memory (9301)
m = read(LOOP + "9301_gpu_ondevice_mem_latency/SUMMARY.md")
sec = m.split("## Peak memory (GiB), fp32")[1].split("## Latency")[0]
mem = {}
for l in sec.split("\n"):
    if l.startswith("|") and not l.startswith("| model") and not l.startswith("|---"):
        c = [x.strip() for x in l.strip("|").split("|")]
        mem[c[0]] = [float(x) for x in c[2:]]
Rs = [256, 512, 768, 1024, 1536, 2048]


def tex_mem():
    s = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{\label{tab:app_mem}Exploratory peak GPU memory (GiB) of a batch-one forward pass in FP32 on an RTX 3090 (\texttt{torch.cuda.max\_memory\_allocated}), for a $256\times256$ input bilinearly upsampled to the resolution $R$. This is a proxy for on-device memory and was not confirmed on the Jetson Nano. Copied by script from package 9301.}",
         r"\begin{tabular}{lrrrrrr}", r"\hline", "Model & " + " & ".join("$R{=}%d$" % r for r in Rs) + r" \\", r"\hline"]
    for k in ["pbe"] + BASE:
        s.append(MN[k] + " & " + " & ".join("%.3f" % v for v in mem[k]) + r" \\")
    s += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(s)


facts = [(R, mem["pbe"][i], [MN[b] for b in BASE if mem[b][i] > mem["pbe"][i]]) for i, R in enumerate(Rs)]

# ---- scale shift (9302/9303/9311-9314), resize-back 9322
SS = {"PH2": "9302_gpu_ondevice_upsample_acc_ph2", "kvasir": "9303_gpu_ondevice_upsample_acc_kvasir",
      "etis": "9311_gpu_upsample_repro_etis", "cvc": "9312_gpu_upsample_repro_cvc",
      "isic17": "9313_gpu_upsample_repro_isic17", "isic18": "9314_gpu_upsample_repro_isic18"}


def scale_table(path):
    t = read(path)
    rows = list(tables("\n## x \n" + t).values())[0][1]
    ch = float(re.search(r"max=([\d.]+)", t).group(1))
    return rows, ch


srows = []
for d in DS:
    rows, ch = scale_table(LOOP + SS[d] + "/SUMMARY.md")
    coll = sum(1 for b in BASE if rows[b]["R=2048"] <= ch + 0.05)
    rb = None
    if d in ("kvasir", "etis", "cvc"):
        r2, c2 = scale_table(LOOP + "9322_gpu_upsample_resizeback/" + d + "/SUMMARY.md")
        bl = [r2[b]["R=2048"] for b in BASE]
        rb = (r2["pbe"]["R=2048"], min(bl), max(bl), sum(1 for b in BASE if r2[b]["R=2048"] <= c2 + 0.05))
    srows.append((d, ch, rows["pbe"]["R=256"], rows["pbe"]["R=2048"], coll, rb))


def tex_scale():
    s = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{\label{tab:app_scale}Exploratory scale-shift analysis on the validation split of split~1. The $256\times256$ validation image is bilinearly upsampled to $R=2048$, the network is applied, and the probability map is area-downsampled to $256\times256$ before thresholding at 0.5. Chance is the larger of the all-background and all-foreground IoU of the validation set. ``Collapsed'' counts the baselines (of ten) with IoU at most chance${}+0.05$. The last column is the test-time resize-back control, in which the upsampled image is area-downsampled to $256\times256$ before the network is applied (run for the three colonoscopy datasets only); it lists the PBE-Net IoU and the range of the ten baseline IoUs at $R=2048$. Copied by script from packages 9302, 9303, 9311--9314 and 9322.}",
         r"\begin{tabular}{lccccc}", r"\hline",
         r"Dataset & Chance & PBE-Net, $R{=}256$ & PBE-Net, $R{=}2048$ & Baselines collapsed & Resize-back, $R{=}2048$ \\", r"\hline"]
    for d, ch, a, b, c, rb in srows:
        rbs = "--" if rb is None else "%.3f; baselines %.3f--%.3f" % (rb[0], rb[1], rb[2])
        s.append("%s & %.3f & %.3f & %.3f & %d/10 & %s \\\\" % (NAME[d], ch, a, b, c, rbs))
    s += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(s)


# ---- noise (9501-9506)
NO = {"etis": "9501_cpu_degrade_etis", "PH2": "9502_cpu_degrade_PH2", "cvc": "9503_cpu_degrade_cvc",
      "kvasir": "9504_cpu_degrade_kvasir", "isic17": "9505_cpu_degrade_isic17", "isic18": "9506_cpu_degrade_isic18"}
CONDS = [("gauss_s04", r"Gaussian, $\sigma{=}0.4$"), ("poisson_p1", "Poisson, peak 1"),
         ("sp_50", r"Salt-and-pepper, 50\%")]
nrows = {}
for d in DS:
    t = read(LOOP + NO[d] + "/SUMMARY.md")
    lines = t.split("\n")
    rows = {}
    for i, l in enumerate(lines):
        if l.startswith("| model | identity"):
            hdr = [c.strip() for c in l.strip("|").split("|")][1:]
            j = i + 2
            while lines[j].startswith("|"):
                c = [x.strip() for x in lines[j].strip("|").split("|")]
                rows[c[0]] = dict(zip(hdr, [float(x) for x in c[1:]]))
                j += 1
    ver = {}
    for l in lines:
        mm = re.match(r"\| (\S+) \| (\d+) \| (\d+) \| (\d+) \| (True|False) \| (\w+) \|", l)
        if mm:
            ver[mm.group(1)] = int(mm.group(2))
    nrows[d] = (rows, ver)


def tex_noise():
    s = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{\label{tab:app_noise}Exploratory acquisition-degradation analysis (CPU, split~1 validation, at most 40 strided validation images per dataset). Each cell gives the PBE-Net IoU and, in parentheses, the number of the ten baselines whose IoU fell to at most chance${}+0.05$ from a healthy value on the clean images. Copied by script from packages 9501--9506.}",
         r"\begin{tabular}{lccc}", r"\hline", "Dataset & " + " & ".join(c[1] for c in CONDS) + r" \\", r"\hline"]
    for d in DS:
        rows, ver = nrows[d]
        s.append(NAME[d] + " & " + " & ".join("%.3f (%d)" % (rows["pbe"][c[0]], ver[c[0]]) for c in CONDS) + r" \\")
    s += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(s)


with open(OUT, "w", encoding="utf8") as f:
    f.write("% GENERATED by tools/gen_appendix_tables.py from the SUMMARY.md files; do not edit by hand.\n")
    for t in (tex_compress(), tex_mem(), tex_scale(), tex_noise()):
        f.write(t + "\n\n")
print("wrote", OUT)
print("--- compression rows (dataset, orig, (w4,above), (w3,above), (l50,above), (w4pc,above))")
for r in rowsA:
    print(r)
print("--- memory: R, pbe, baselines higher than pbe")
for x in facts:
    print(x)
print("--- scale")
for x in srows:
    print(x)
print("--- noise gauss_s01 (pbe, fell):", {d: (nrows[d][0]["pbe"]["gauss_s01"], nrows[d][1]["gauss_s01"]) for d in DS})
print("--- noise strongest fell counts:", {d: [nrows[d][1][c[0]] for c in CONDS] for d in DS})
