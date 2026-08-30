# fig_ledger_aggregation.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
})

def add_box(ax, xy, w, h, text="", fc="0.96", ec="0.15", lw=1.0, fontsize=11,
            ha="center", va="center", weight="normal", ls="-"):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)
    ax.add_patch(rect)
    if text:
        ax.text(xy[0] + w/2, xy[1] + h/2, text, ha=ha, va=va,
                fontsize=fontsize, weight=weight)
    return rect

def add_circle(ax, center, r, text="", fc="0.96", ec="0.15", lw=1.0, fontsize=11):
    c = Circle(center, r, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(c)
    if text:
        ax.text(center[0], center[1], text, ha="center", va="center", fontsize=fontsize)
    return c

def add_arrow(ax, p1, p2, color="0.3", lw=1.4, ms=18):
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                          linewidth=lw, color=color)
    ax.add_patch(arr)
    return arr

fig, ax = plt.subplots(figsize=(15, 5.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# -------- Panel layout --------
x0 = 0.03
panel_w = 0.205
gap = 0.03
y0 = 0.18
panel_h = 0.70

panels = []
for i in range(4):
    px = x0 + i*(panel_w + gap)
    panels.append((px, y0, panel_w, panel_h))
    add_box(ax, (px, y0), panel_w, panel_h, fc="1.0", ec="0.4", lw=0.8)

titles = [
    "(a) Interaction pots",
    "(b) Potwise transfer ledger",
    "(c) Feature aggregation",
    "(d) Observed discrepancy"
]
for (px, py, pw, ph), t in zip(panels, titles):
    ax.text(px + 0.01, py + ph + 0.03, t, ha="left", va="bottom",
            fontsize=13, weight="bold")

# -------- Panel (a): interaction pots --------
px, py, pw, ph = panels[0]
ys = [py + ph*0.78, py + ph*0.56, py + ph*0.34, py + ph*0.12]
pot_labels = [
    r"$u_1=\{i_1,i_2\}$",
    r"$u_2=\{i_1,i_3,i_4\}$",
    r"$\vdots$",
    r"$u_M=\{i_2,i_3\}$"
]
circ_texts = [r"$u_1$", r"$u_2$", r"$\cdots$", r"$u_M$"]

for yy, clab, plab in zip(ys, circ_texts, pot_labels):
    if clab != r"$\cdots$":
        add_circle(ax, (px + 0.055, yy), 0.018, text=clab, fc="0.94", ec="0.25", lw=0.9)
    else:
        ax.text(px + 0.055, yy, r"$\vdots$", ha="center", va="center", fontsize=16)
    ax.text(px + 0.11, yy, plab, ha="left", va="center", fontsize=14)

ax.text(px + pw/2, py + 0.03, r"$M$ interaction pots", ha="center", va="bottom", fontsize=12)

# -------- Panel (b): transfer ledger --------
px, py, pw, ph = panels[1]
# matrix frame
mx, my = px + 0.055, py + 0.16
mw, mh = pw - 0.09, ph - 0.28
add_box(ax, (mx, my), mw, mh, fc="0.99", ec="0.2", lw=1.0)

cols = 4
rows = 4
for j in range(1, cols):
    x = mx + j*mw/cols
    ax.add_line(Line2D([x, x], [my, my+mh], color="0.35", linewidth=0.8))
for i in range(1, rows):
    y = my + i*mh/rows
    ax.add_line(Line2D([mx, mx+mw], [y, y], color="0.35", linewidth=0.8))

col_headers = [r"$1$", r"$2$", r"$\cdots$", r"$d$"]
row_headers = [r"$u_1$", r"$u_2$", r"$\vdots$", r"$u_M$"]

for j, txt in enumerate(col_headers):
    xx = mx + (j + 0.5)*mw/cols
    ax.text(xx, my + mh + 0.04, txt, ha="center", va="center", fontsize=13)

ax.text(mx + mw/2, my + mh + 0.085, r"features $i\in[d]$", ha="center", va="center", fontsize=12)

for i, txt in enumerate(row_headers):
    yy = my + mh - (i + 0.5)*mh/rows
    ax.text(mx - 0.03, yy, txt, ha="right", va="center", fontsize=13)

cell_text = [
    [r"$T_{1,u_1}$", r"$T_{2,u_1}$", r"$\cdots$", r"$T_{d,u_1}$"],
    [r"$T_{1,u_2}$", r"$T_{2,u_2}$", r"$\cdots$", r"$T_{d,u_2}$"],
    [r"$\vdots$", r"$\vdots$", r"$\ddots$", r"$\vdots$"],
    [r"$T_{1,u_M}$", r"$T_{2,u_M}$", r"$\cdots$", r"$T_{d,u_M}$"],
]
for i in range(rows):
    for j in range(cols):
        xx = mx + (j + 0.5)*mw/cols
        yy = my + mh - (i + 0.5)*mh/rows
        ax.text(xx, yy, cell_text[i][j], ha="center", va="center", fontsize=13)

ax.text(px + pw/2, py + 0.085,
        r"Ledger $T$: column sums are zero (pot conservation)",
        ha="center", va="center", fontsize=11)

# -------- Panel (c): aggregation --------
px, py, pw, ph = panels[2]
ax.text(px + pw/2, py + ph - 0.06, r"Sum over pots for each feature $i$",
        ha="center", va="center", fontsize=12)

vx, vy = px + 0.085, py + 0.18
vw, vh = 0.085, 0.43
add_box(ax, (vx, vy), vw, vh, fc="0.99", ec="0.2", lw=1.0)

nslots = 4
for i in range(1, nslots):
    y = vy + i*vh/nslots
    ax.add_line(Line2D([vx, vx+vw], [y, y], color="0.35", linewidth=0.8))

vec_entries = [
    r"$\sum_u T_{1,u}$",
    r"$\sum_u T_{2,u}$",
    r"$\vdots$",
    r"$\sum_u T_{d,u}$"
]
for i, txt in enumerate(vec_entries):
    yy = vy + vh - (i + 0.5)*vh/nslots
    ax.text(vx + vw/2, yy, txt, ha="center", va="center", fontsize=13)

ax.text(vx + vw + 0.045, vy + vh/2, r"$=\,b-a$", ha="left", va="center", fontsize=17)

ax.text(px + pw/2, py + 0.085,
        r"Linear aggregation map $A(T)=b-a$",
        ha="center", va="center", fontsize=12)

# -------- Panel (d): observed discrepancy --------
px, py, pw, ph = panels[3]
ax.text(px + pw/2, py + ph - 0.06, r"Feature-level quantity retained after aggregation",
        ha="center", va="center", fontsize=11)

ox, oy = px + 0.09, py + 0.20
ow, oh = 0.08, 0.38
add_box(ax, (ox, oy), ow, oh, fc="0.99", ec="0.2", lw=1.0)
for i in range(1, 4):
    y = oy + i*oh/4
    ax.add_line(Line2D([ox, ox+ow], [y, y], color="0.35", linewidth=0.8))

obs_entries = [r"$b_1-a_1$", r"$b_2-a_2$", r"$\vdots$", r"$b_d-a_d$"]
for i, txt in enumerate(obs_entries):
    yy = oy + oh - (i + 0.5)*oh/4
        # feature labels
    ax.text(ox - 0.03, yy, f"{i+1}" if i < 2 else (r"$\vdots$" if i == 2 else r"$d$"),
            ha="right", va="center", fontsize=12)
    ax.text(ox + ow/2, yy, txt, ha="center", va="center", fontsize=13)

# info loss note
add_box(ax, (px + 0.125, py + 0.14), 0.065, 0.10,
        text="pot index\nremoved", fc="0.92", ec="0.25", lw=0.8, fontsize=11)

add_box(ax, (px + 0.05, py + 0.03), pw - 0.10, 0.08,
        text="Different ledgers may map to the same observed vector $b-a$.",
        fc="0.94", ec="0.30", lw=0.8, fontsize=11)

# -------- Arrows between panels --------
for i in range(3):
    x_left = panels[i][0] + panels[i][2]
    x_right = panels[i+1][0]
    y_mid = y0 + panel_h/2
    add_arrow(ax, (x_left + 0.01, y_mid), (x_right - 0.01, y_mid), color="0.35", lw=1.5, ms=20)

# -------- Caption --------
caption = (
    r"$\bf{Figure\ 1.\ From\ transfer\ ledger\ to\ aggregate\ discrepancy.}$ "
    r"Two attribution rules may allocate the same interaction pots differently. "
    r"The transfer ledger $T$ records the potwise redistribution between the two rules; "
    r"each column sums to zero because both rules conserve the same pot value. "
    r"Feature aggregation applies the linear map $A$, summing over pots incident to each feature, "
    r"and produces the observable discrepancy $b-a$. The key information loss is that the pot index is discarded: "
    r"distinct ledgers can yield the same aggregate vector."
)
ax.text(0.03, 0.06, caption, ha="left", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("fig_ledger_aggregation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_ledger_aggregation.png")