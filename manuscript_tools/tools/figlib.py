"""Small matplotlib drawing helpers shared by draw_fig1.py and draw_fig2.py (D6 redraw).
Coordinates are in PDF points, y axis pointing up."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle

matplotlib.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# palette taken from the original figures
C = dict(
    ebpb_fc="#FFD8B1", ebpb_ec="#B8541A",
    basic_fc="#BDD7EE", basic_ec="#5B7FAE",
    bott_fc="#9DC3E6", bott_ec="#2E75B6",
    conv_fc="#ECE9C8", gray_fc="#CFCFCF", prcm_fc="#DDD0E6",
    pool_fc="#FFF2B3", pool_ec="#5BAA3A", line="#111111", note="#444444",
)


def canvas(w, h):
    fig = plt.figure(figsize=(w / 72.0, h / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x0, y0, x1, y1, fc="white", ec="black", lw=1.2, ls="-", r=3, z=3):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=%g" % r,
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, fs=12, ha="center", va="center", rot=0, color="black", z=6, style="normal", weight="normal"):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, rotation=rot, color=color, zorder=z,
                   fontstyle=style, fontweight=weight, linespacing=1.05)


def line(ax, pts, lw=1.2, ls="-", color="black", z=2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, lw=lw, ls=ls, color=color, zorder=z, solid_capstyle="butt")


def arrow(ax, pts, lw=1.2, ls="-", color="black", z=2, head=7):
    """polyline whose last segment ends in an arrow head"""
    if len(pts) > 2:
        line(ax, pts[:-1], lw, ls, color, z)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    line(ax, [(x0, y0), (x1, y1)], lw, ls, color, z)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_length=%g,head_width=%g" % (head / 14.0, head / 28.0),
                                color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=14, ls="-"))


def node(ax, x, y, sym, r=7, fs=11, z=5):
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=1.2, zorder=z))
    ax.plot([x - r * .6, x + r * .6], [y, y], color="black", lw=1.2, zorder=z + 1)
    if sym == "+":
        ax.plot([x, x], [y - r * .6, y + r * .6], color="black", lw=1.2, zorder=z + 1)


def dot(ax, x, y, r=2.6, z=6):
    ax.add_patch(Circle((x, y), r, fc="black", ec="black", zorder=z))


def cube(ax, x, y, w, h, d, fc="#BDD7EE", z=3):
    """front face at (x,y)-(x+w,y+h), depth offset d towards upper right"""
    side = "#9AAFC9"; top = "#D7E6F5"
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)],
                         fc=side, ec="#40536D", lw=0.9, zorder=z))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d), (x + w, y + h)],
                         fc=top, ec="#40536D", lw=0.9, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="#40536D", lw=0.9, zorder=z + .1))


def save(fig, path):
    fig.savefig(path, format="pdf")
