# fig_forest_vs_cycle.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
})

def add_box(ax, xy, w, h, text="", fc="0.98", ec="0.2", lw=1.0,
            fontsize=11, weight="normal"):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    if text:
        ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight)
    return rect

def add_square_node(ax, center, label, size=0.038, fc="0.93", ec="0.15", lw=1.0):
    x, y = center
    rect = Rectangle((x - size/2, y - size/2), size, size,
                     facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=12)
    return rect

def add_circle_node(ax, center, label, r=0.020, fc="0.96", ec="0.15", lw=1.0):
    circ = Circle(center, r, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(circ)
    ax.text(center[0], center[1], label, ha="center", va="center", fontsize=12)
    return circ

def add_edge(ax, p1, p2, color="0.15", lw=1.2, ls="-"):
    ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, linestyle=ls))

def add_arrow(ax, p1, p2, color="0.35", lw=1.2, ms=16, ls="-"):
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                          linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(arr)

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Outer panels
left = (0.03, 0.18, 0.45, 0.68)
right = (0.52, 0.18, 0.45, 0.68)

for panel in [left, right]:
    add_box(ax, (panel[0], panel[1]), panel[2], panel[3], fc="1.0", ec="0.4", lw=0.8)

ax.text(left[0] + 0.015, left[1] + left[3] + 0.03,
        "(a) Forest (acyclic)  \u2192  identifiable",
        ha="left", va="bottom", fontsize=14, weight="bold")
ax.text(right[0] + 0.015, right[1] + right[3] + 0.03,
        "(b) Cycle  \u2192  non-identifiable",
        ha="left", va="bottom", fontsize=14, weight="bold")

# ---------------- LEFT PANEL: forest ----------------
lx, ly, lw_, lh = left

ax.text(lx + 0.14, ly + lh - 0.07, "Feature–interaction incidence graph",
        ha="center", va="center", fontsize=12)

# Nodes
f1 = (lx + 0.08, ly + 0.46)
f2 = (lx + 0.18, ly + 0.46)
f3 = (lx + 0.28, ly + 0.46)
f4 = (lx + 0.38, ly + 0.46)

u1 = (lx + 0.11, ly + 0.24)
u2 = (lx + 0.23, ly + 0.24)
u3 = (lx + 0.35, ly + 0.24)

for c, lbl in zip([f1, f2, f3, f4], [r"$1$", r"$2$", r"$3$", r"$4$"]):
    add_square_node(ax, c, lbl)

for c, lbl in zip([u1, u2, u3], [r"$u_1$", r"$u_2$", r"$u_3$"]):
    add_circle_node(ax, c, lbl)

# edges (forest)
add_edge(ax, f1, u1)
add_edge(ax, f2, u1)
add_edge(ax, f2, u2)
add_edge(ax, f3, u2)
add_edge(ax, f4, u3)

# theorem consequence boxes
add_box(ax, (lx + 0.09, ly + 0.08), 0.12, 0.08, text=r"$\ker A=\{0\}$",
        fc="0.94", ec="0.2", lw=0.9, fontsize=14)
add_box(ax, (lx + 0.24, ly + 0.08), 0.18, 0.08,
        text="unique compatible ledger", fc="0.94", ec="0.2", lw=0.9, fontsize=11)

ax.text(lx + lw_/2, ly + 0.02,
        "No hidden circulation: aggregate discrepancy determines the full ledger.",
        ha="center", va="bottom", fontsize=11)

# ---------------- RIGHT PANEL: cycle ----------------
rx, ry, rw_, rh = right

ax.text(rx + 0.14, ry + rh - 0.07, "Feature–interaction incidence graph",
        ha="center", va="center", fontsize=12)

g1 = (rx + 0.08, ry + 0.46)
g2 = (rx + 0.18, ry + 0.46)
g3 = (rx + 0.28, ry + 0.46)

v12 = (rx + 0.12, ry + 0.22)
v23 = (rx + 0.21, ry + 0.22)
v13 = (rx + 0.30, ry + 0.22)

for c, lbl in zip([g1, g2, g3], [r"$1$", r"$2$", r"$3$"]):
    add_square_node(ax, c, lbl)
for c, lbl in zip([v12, v23, v13], [r"$u_{12}$", r"$u_{23}$", r"$u_{13}$"]):
    add_circle_node(ax, c, lbl)

# cycle edges
add_edge(ax, g1, v12)
add_edge(ax, g2, v12)
add_edge(ax, g2, v23)
add_edge(ax, g3, v23)
add_edge(ax, g1, v13)
add_edge(ax, g3, v13)

# circulation arrows (dashed)
add_arrow(ax, (g1[0], g1[1]-0.03), (v12[0], v12[1]+0.03), color="0.35", ls="--")
add_arrow(ax, (v12[0]+0.025, v12[1]), (g2[0]-0.025, g2[1]-0.03), color="0.35", ls="--")
add_arrow(ax, (g2[0]+0.025, g2[1]-0.03), (v23[0]-0.025, v23[1]+0.03), color="0.35", ls="--")
add_arrow(ax, (v23[0]+0.025, v23[1]), (g3[0]-0.025, g3[1]-0.03), color="0.35", ls="--")
add_arrow(ax, (g3[0]-0.01, g3[1]-0.03), (v13[0]+0.005, v13[1]+0.03), color="0.35", ls="--")
add_arrow(ax, (v13[0]-0.025, v13[1]), (g1[0]+0.01, g1[1]-0.03), color="0.35", ls="--")

# plus/minus small labels near arrows
for x, y, t in [
    (rx + 0.095, ry + 0.35, r"$+$"),
    (rx + 0.15,  ry + 0.35, r"$-$"),
    (rx + 0.195, ry + 0.35, r"$+$"),
    (rx + 0.25,  ry + 0.35, r"$-$"),
    (rx + 0.19,  ry + 0.18, r"$-$"),
    (rx + 0.285, ry + 0.18, r"$+$"),
]:
    ax.text(x, y, t, ha="center", va="center", fontsize=14, color="0.25")

add_box(ax, (rx + 0.09, ry + 0.08), 0.14, 0.08, text=r"$\dim \ker A \geq 1$",
        fc="0.94", ec="0.2", lw=0.9, fontsize=14)
add_box(ax, (rx + 0.25, ry + 0.08), 0.16, 0.08,
        text="many compatible ledgers", fc="0.94", ec="0.2", lw=0.9, fontsize=11)

ax.text(rx + rw_/2, ry + 0.02,
        "A nonzero circulation can be added without changing the aggregate discrepancy $b-a$.",
        ha="center", va="bottom", fontsize=11)

# -------- Caption --------
caption = (
    r"$\bf{Figure\ 2.\ Graph\ criterion\ for\ identifiability.}$ "
    r"The aggregation map $A$ observes only featurewise row sums of the transfer ledger. "
    r"Its kernel is the circulation space of the feature–interaction incidence graph. "
    r"If the graph is a forest, then $\ker A=\{0\}$ and the ledger is uniquely identified by the aggregate discrepancy. "
    r"If the graph contains a cycle, then nonzero hidden circulations exist, so multiple distinct ledgers map to the same observed vector."
)
ax.text(0.03, 0.06, caption, ha="left", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("fig_forest_vs_cycle.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_forest_vs_cycle.png")