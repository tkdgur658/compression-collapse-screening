"""Jetson column of Table 2 from results/jetson/JETSON_RESULT.json only (PREREG_JETSON).
Cell = median [IQR] in ms (FP16 TensorRT, deployment graph); a model without a reportable block gets n.m. with a
footnote mark per reason. LEGACY comment lines of tables/tb_eff.tex are kept unchanged.
Default is a dry run that writes the preview to --out. --apply rewrites tables/tb_eff.tex and refuses when no model has a
reportable value (an all-n.m. column would reverse the Pass 22 deletion, which is a study lead decision).
Usage: python -I -B tools/gen_jetson_col.py [--out PREVIEW.tex] [--apply]"""
import sys, os, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.dirname(HERE)
RES = os.path.join(os.path.dirname(MC), 'iso', 's4', 'jetson', 'JETSON_RESULT.json')
TB = os.path.join(MC, 'tables', 'tb_eff.tex')
ROWS = [('PBE-Net', 'pbe'), ('MAResUNet', 'maresunet'), ('UTANet', 'utanet'), ('MK-UNet', 'mk_unet'), ('MHorUNet', 'mhorunet'),
        ('TinyU-Net', 'tinyunet'), ('UCMNet', 'ucmnet'), ('LB-UNet', 'lbunet'), ('CMUNeXt', 'cmunext'), ('EGE-UNet', 'egeunet'),
        ('MALUNet', 'malunet')]


def reason_class(r):
    s = r.get('reason', '')
    if s.startswith('ONNX opset 11 export failed'): return 'export'
    if 'ONNX parser failed' in s: return 'parser'
    if s.startswith('unstable'): return 'clock'
    return 'other'


NOTE = {'export': 'operator not exportable to ONNX opset 11',
        'parser': 'ONNX graph not parsed by TensorRT %s',
        'clock': 'clock state changed during the block in three attempts',
        'other': 'not measured'}


def build(d):
    ms = d['models']; marks = {}; cells = {}
    for _, k in ROWS:
        r = ms[k]
        if r['status'] == 'measured':
            cells[k] = '%.1f [%.1f]' % (r['median_ms'], r['iqr_ms'])
        else:
            c = reason_class(r); marks.setdefault(c, 'abcd'[len(marks)])
            cells[k] = 'n.m.\\textsuperscript{%s}' % marks[c]
    trt = next((ms[k].get('tensorrt_version') for _, k in ROWS if ms[k].get('tensorrt_version')), '?')
    foot = ' '.join('\\textsuperscript{%s}%s.' % (m, (NOTE[c] % trt) if '%s' in NOTE[c] else NOTE[c]) for c, m in marks.items())
    return cells, foot, trt, sum(ms[k]['status'] == 'measured' for _, k in ROWS)


def render(src, cells, foot, trt):
    out = src.replace('colspec={X[l,1.3] X[c,1] X[c,1] X[c,1]}', 'colspec={X[l,1.3] X[c,1] X[c,1] X[c,0.8] X[c,1.2]}')
    out = out.replace(' Model & Params (M) & FLOPs (G) & RTX (ms) \\\\', ' Model & Params (M) & FLOPs (G) & RTX (ms) & Jetson (ms) \\\\')
    for disp, k in ROWS:
        out = re.sub(r'(?m)^( %s & [^\n]*?) \\\\$' % re.escape(disp), lambda mm: '%s & %s \\\\' % (mm.group(1), cells[k]), out, count=1)
    cond = ('Jetson: NVIDIA Jetson Nano, JetPack L4T R32.7.6, TensorRT %s, FP16 engine of the deployment graph exported to ONNX '
            'opset 11, batch size one, $256\\times256$ input already on the device. After 10 warm-up runs, 50 timed runs give '
            'the GPU compute time, reported as median [interquartile range] in ms. Clocks were not fixed and were recorded before '
            'and after each model. %s' % (trt, foot))
    out = out.replace('\n}\n\\end{table}', ' ' + cond + '\n}\n\\end{table}', 1)
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    d = json.load(open(RES))
    cells, foot, trt, n_meas = build(d)
    src = open(TB, encoding='utf-8', newline='').read()
    new = render(src, cells, foot, trt)
    print('measured models:', n_meas, '| cells:', cells, '| footnote:', foot)
    if '--out' in a:
        open(a[a.index('--out') + 1], 'w', encoding='utf-8', newline='').write(new); print('preview written')
    if '--apply' in a:
        if n_meas == 0:
            sys.exit('no model has a reportable Jetson value: tb_eff.tex not changed (study lead decision needed)')
        open(TB, 'w', encoding='utf-8', newline='').write(new); print('tb_eff.tex written')
