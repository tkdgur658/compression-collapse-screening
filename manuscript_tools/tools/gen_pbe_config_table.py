"""Pass 32 (journal review 2, item A-6, [internal decision] = [id], 2026-10-16): configuration table of PBE-Net
-> tables/tb_app_pbe_config.tex, read from the training code (read-only).

Sources (sha256 checked, the script stops on a mismatch):
  code/pbe_seg/models/pbe/src/net.py   (de4dcb9c...) PBENet, Base, stage, RepConv, EBPB, PRCM, DWSepBottleneck
  code/pbe_seg/models/pbe/adapter.py   (23e061bb...) DEFAULT_CFG
  code/pbe_seg/common/data.py          (c97be2e1...) training augmentation
Every value printed in the table is asserted as a literal string of these files below.
Run:  python -s tools/gen_pbe_config_table.py
"""
import hashlib, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", "reference", "pbe_seg_98f478d"))
OUT = os.path.join(HERE, "..", "tables", "tb_app_pbe_config.tex")
FILES = {"models/pbe/src/net.py": "de4dcb9c59d4a49c96553bced07621635ed0bde57e23bb6b2bad39d754d33562",
         "models/pbe/adapter.py": "23e061bbb1a88cf642ed7ed1dffd1149829499361d9aaeaab4a4d29b46a842e6",
         "common/data.py": "c97be2e1f908fac83041b1b0745315f1a9944c8b7cd9e585a97ae8a9267b46c0"}
# (file, literal that must occur)
CHECKS = [
    ("models/pbe/adapter.py", "'c_list': [128, 192, 256, 384, 512]"),
    ("models/pbe/adapter.py", "'skip': 'predictive', 'block': 'ebpb'"),
    ("models/pbe/adapter.py", "'lambda_s': 0.1, 'band_kernel': 3"),
    ("models/pbe/src/net.py", "self.encoder1 = stage(3, c[0], band=True); self.encoder2 = stage(c[0], c[1], band=True)"),
    ("models/pbe/src/net.py", "layers = [nn.Conv2d(in_ch, out_ch, 1, bias=False)] if in_ch != out_ch else []"),
    ("models/pbe/src/net.py", "self.conv_kxk = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels, bias=False)"),
    ("models/pbe/src/net.py", "self.conv_1x1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)"),
    ("models/pbe/src/net.py", "self.bn_identity = nn.BatchNorm2d(channels)"),
    ("models/pbe/src/net.py", "self.pool4 = nn.AvgPool2d(r, r)"),
    ("models/pbe/src/net.py", "def __init__(self, c_list=(128, 192, 256, 384, 512), skip='predictive', block='ebpb', num_classes=1, r=4):"),
    ("models/pbe/src/net.py", "nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),"),
    ("models/pbe/src/net.py", "nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))"),
    ("models/pbe/src/net.py", "mode='bilinear', align_corners=False"),
    ("models/pbe/src/net.py", "def __init__(self, channels, num_basis=8, dropout_rate=0.5):"),
    ("models/pbe/src/net.py", "return x * self.fuser(coeff).sigmoid()[..., None, None]"),
    ("models/pbe/src/net.py", "self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False)"),
    ("models/pbe/src/net.py", "self.pw = nn.Conv2d(ch, ch, 1, bias=False)"),
    ("models/pbe/src/net.py", "self.proj1 = nn.Conv2d(c[1], c[0], 1, bias=False)"),
    ("models/pbe/src/net.py", "nn.init.zeros_(m.weight)"),
    ("models/pbe/src/net.py", "self.shead1 = nn.Conv2d(c[0], 1, 1); self.shead2 = nn.Conv2d(c[1], 1, 1); self.shead3 = nn.Conv2d(c[2], 1, 1)"),
    ("models/pbe/src/net.py", "self.decoder1 = stage(c[4], c[3]); self.decoder2 = stage(c[3], c[2])"),
    ("models/pbe/src/net.py", "e1 = F.max_pool2d(b.encoder1(x), 2)"),
    ("common/data.py", "ang = random.uniform(0.0, 360.0)"),
]
ROWS = [
    ("Channel widths", "128, 192, 256, 384, and 512 for encoder stages 1 to 5, mirrored by the four decoder stages"),
    ("Stage unit", r"$1\times1$ convolution when the width changes, RepConv, the feature block at encoder stages 1 and 2, channel gate"),
    ("RepConv", r"$3\times3$ depthwise convolution with batch normalization, $1\times1$ depthwise branch with batch normalization, identity batch normalization, ReLU, with the branches merged in the deployed graph"),
    ("Feature block", r"average pooling with stride $r=4$, bilinear upsampling, high-pass residual kept, $\psi$ on the pooled map"),
    (r"$\psi$", r"$3\times3$ depthwise convolution, batch normalization, ReLU, $1\times1$ convolution, batch normalization, ReLU"),
    ("Channel gate", r"global average, projection on 8 learned basis vectors, dropout 0.5 on the 8 coefficients, linear map to the channels, sigmoid"),
    ("Bottleneck", r"after encoder stage 5: $7\times7$ depthwise convolution, batch normalization, ReLU, $1\times1$ convolution, batch normalization, ReLU"),
    ("Downsampling", r"$2\times2$ max pooling after encoder stages 1 to 4"),
    ("Predictive skips", r"stages 1 to 3, bias-free $1\times1$ projections of the next stage, zero-initialized, bilinear upsampling"),
    ("Boundary heads", r"$1\times1$ convolutions at stages 1 to 3, training only, matching weight $\lambda_s = 0.1$, $3\times3$ morphological gradient"),
    ("Augmentation", r"horizontal and vertical flips ($p=0.5$ each), rotation with $p=0.5$ by an angle uniform in $[0, 360)$ degrees"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    txt = {}
    for f, h in FILES.items():
        fp = os.path.join(CODE, f)
        if sha(fp) != h:
            sys.exit("sha256 mismatch: %s %s" % (f, sha(fp)))
        txt[f] = io.open(fp, encoding="utf8").read()
    for f, lit in CHECKS:
        assert lit in txt[f], (f, lit)
    body = "\n".join(r" %s & %s \\" % r for r in ROWS)
    tex = r"""%% GENERATED by tools/gen_pbe_config_table.py from code/pbe_seg (net.py sha256 %s, adapter.py %s, data.py %s), do not edit by hand.
\begin{table}[tp]
\centering
\caption{\label{tab:pbe_config}Configuration of PBE-Net as set in the training code (model and augmentation of protocol v1).}
\footnotesize
\begin{tblr}{colspec={X[1,l] X[3.6,l]}, width=\linewidth, colsep=2pt, rowsep=0.5pt}
\hline
 Item & Setting \\
\hline
%s
\hline
\end{tblr}
\end{table}
""" % (FILES["models/pbe/src/net.py"][:8], FILES["models/pbe/adapter.py"][:8], FILES["common/data.py"][:8], body)
    io.open(OUT, "w", encoding="utf8", newline="\n").write(tex)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
