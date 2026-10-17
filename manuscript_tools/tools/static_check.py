"""Static LaTeX checks (no TeX installation available): labels/refs, cites vs main.bib, braces, environments, CHECK count."""
import re, glob, io, os
from collections import Counter

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
tex = {p: io.open(p, encoding="utf8").read() for p in glob.glob("**/*.tex", recursive=True)}
labels, refs, cites = set(), set(), set()
nchk = 0
for p, t in tex.items():
    lines = t.split("\n")
    nchk += sum(1 for l in lines if re.match(r"\s*%\s*CHECK", l))
    t2 = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in lines)
    labels |= set(re.findall(r"\\label\{([^}]+)\}", t2))
    refs |= set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", t2))
    for c in re.findall(r"\\cite\{([^}]+)\}", t2):
        cites |= {x.strip() for x in c.split(",")}
    d = t2.count("{") - t2.count("}") - len(re.findall(r"\\[{}]", t2)) * 0
    esc_o = len(re.findall(r"\\\{", t2)); esc_c = len(re.findall(r"\\\}", t2))
    d = (t2.count("{") - esc_o) - (t2.count("}") - esc_c)
    if d:
        print("brace imbalance", p, d)
    b = Counter(re.findall(r"\\begin\{([^}]+)\}", t2)); e = Counter(re.findall(r"\\end\{([^}]+)\}", t2))
    if b != e:
        print("env mismatch", p, b - e, e - b)
    for bad in ("\t", "\x08", "\x0c"):
        if bad in t:
            print("control char in", p, repr(bad))
    if re.search(r"\r(?!\n)", t):
        print("lone CR in", p)
bib = io.open("main.bib", encoding="utf8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("undefined refs:", sorted(refs - labels))
print("missing cites:", sorted(cites - keys))
print("unused labels:", sorted(labels - refs))
print("CHECK comments:", nchk)
