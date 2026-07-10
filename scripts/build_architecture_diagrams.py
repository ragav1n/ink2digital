#!/usr/bin/env python3
"""Build presentation architecture diagrams (old pipeline vs new VLM system).

Draws two left-to-right flow diagrams and saves them as high-resolution JPGs:
    figures/architecture_old.jpg  -- the Phase 1-3 pipeline of specialists
    figures/architecture_new.jpg  -- the deployed single-VLM system (infer.py)

Everything here mirrors the real, deployed system:
  * The YOLOv8 detector is RETAINED in the new system; the VLM replaces the two
    OCR specialists (TrOCR + TAMER/Pix2Tex) and the Reptile adaptation step.
  * The per-professor QLoRA adapter is optional (~0.1% of params).

Run:  python3 scripts/build_architecture_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette ---------------------------------------------------------------
INK = "#23303B"          # primary text
MUTED = "#5A6B79"        # secondary text
ARROW = "#56616B"
RED = "#C0392B"          # failure annotations
GREEN_TX = "#2E7D43"     # adapter / success accent

SLIDE = dict(fc="#EDF2F7", ec="#6B8092")     # input / output slide
DETECT = dict(fc="#FFF1D6", ec="#D99B2B")    # detector
READER = dict(fc="#E4EEF8", ec="#5E86A8")    # OCR specialists (old)
VLM = dict(fc="#DBEEDD", ec="#3F9A52")       # the vision-language model (new)
RENDER = dict(fc="#ECE3F4", ec="#8A6BB0")    # render / overlay
BADGE = dict(fc="#EAF6EC", ec="#3F9A52")     # optional adapter badge

FONT = {"family": "DejaVu Sans"}


def box(ax, cx, cy, w, h, title, subtitle=None, *, style, title_size=12,
        sub_size=8.6, title_color=INK):
    """Rounded box centred at (cx, cy) with a bold title and optional subtitle."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.8, facecolor=style["fc"], edgecolor=style["ec"],
        mutation_aspect=1.0, zorder=2,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(cx, cy + 0.20, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, **FONT)
        ax.text(cx, cy - 0.26, subtitle, ha="center", va="center",
                fontsize=sub_size, color=MUTED, **FONT)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, **FONT)


def arrow(ax, x0, y0, x1, y1, *, lw=2.0, style="-|>", color=ARROW, ls="-"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                        linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=1,
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, format="jpg", facecolor="white",
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def build_old(path):
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(7.75, 7.15, "Old architecture — pipeline of specialists  (Phases 1–3)",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK, **FONT)

    ymid = 4.0
    box(ax, 1.5, ymid, 2.1, 1.4, "Input slide", "handwritten\nlecture page", style=SLIDE)
    box(ax, 4.5, ymid, 2.3, 1.4, "YOLOv8", "detect text /\nmath regions", style=DETECT)

    y_up, y_dn = 5.55, 2.45
    box(ax, 8.2, y_up, 3.4, 1.55, "TrOCR (fine-tuned)", "+ Reptile meta-init  →  German text",
        style=READER, title_size=11.5)
    box(ax, 8.2, y_dn, 3.4, 1.55, "TAMER / Pix2Tex", "math regions  →  LaTeX",
        style=READER, title_size=11.5)

    box(ax, 11.6, ymid, 1.9, 1.45, "Render", "typeset\noverlay", style=RENDER)
    box(ax, 13.9, ymid, 1.9, 1.4, "Output slide", "typeset\nreplacement", style=SLIDE,
        title_size=11.5)

    # flow
    arrow(ax, 2.55, ymid, 3.3, ymid)
    arrow(ax, 5.65, ymid + 0.25, 6.45, y_up - 0.2)
    arrow(ax, 5.65, ymid - 0.25, 6.45, y_dn + 0.2)
    arrow(ax, 9.95, y_up - 0.2, 10.6, ymid + 0.25)
    arrow(ax, 9.95, y_dn + 0.2, 10.6, ymid - 0.25)
    arrow(ax, 12.6, ymid, 12.9, ymid)

    # honest failure annotations
    ax.text(4.5, 2.95, "✗ misses handwritten math", ha="center", va="center",
            fontsize=9, color=RED, **FONT)
    ax.text(8.0, 4.45, "✗ Reptile: no gain on target writer\n(33.19% vs 33.26% base)",
            ha="center", va="center", fontsize=8.6, color=RED, linespacing=1.35, **FONT)
    ax.text(8.2, 1.4, "✗ garbage LaTeX on real ink (Pix2Tex CER 284.92%)",
            ha="center", va="center", fontsize=9, color=RED, **FONT)

    ax.text(7.75, 0.55,
            "Three chained specialists + per-line crops — each link broke on real handwriting.",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic", **FONT)
    save(fig, path)


def build_new(path):
    fig, ax = plt.subplots(figsize=(15.5, 6.0))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    ax.text(7.75, 5.6, "New architecture — single vision-language model  (deployed)",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK, **FONT)

    ymid = 3.6
    box(ax, 1.5, ymid, 2.1, 1.4, "Input slide", "handwritten\nlecture page", style=SLIDE)
    box(ax, 4.6, ymid, 2.6, 1.4, "YOLOv8 + ink fallback",
        "regions · NMS · row-stitch", style=DETECT, title_size=11)
    box(ax, 8.5, ymid, 3.6, 1.75, "Qwen3-VL-8B",
        "one model: German text + math\n→ text & LaTeX", style=VLM,
        title_size=13, title_color=GREEN_TX)
    box(ax, 11.9, ymid, 1.85, 1.45, "Render", "typeset\noverlay", style=RENDER)
    box(ax, 14.05, ymid, 1.7, 1.4, "Output slide", "typeset\nreplacement", style=SLIDE,
        title_size=11)

    # flow
    arrow(ax, 2.55, ymid, 3.3, ymid)
    arrow(ax, 5.9, ymid, 6.7, ymid)
    arrow(ax, 10.3, ymid, 10.97, ymid)
    arrow(ax, 12.82, ymid, 13.2, ymid)

    # optional adapter badge under the VLM
    box(ax, 8.5, 1.5, 5.0, 0.82,
        "optional QLoRA adapter", "per-professor · ~0.1% params · 50 crops · ~7 min",
        style=BADGE, title_size=10.5, sub_size=8.2, title_color=GREEN_TX)
    arrow(ax, 8.5, 1.91, 8.5, ymid - 0.92, lw=1.6, color=GREEN_TX, ls=(0, (4, 3)))

    ax.text(7.75, 0.55,
            "One reader, full-region context   —   Deployment: V100 CUDA (8-bit)   |   "
            "Apple Silicon MLX (8-bit, offline)",
            ha="center", va="center", fontsize=10.5, color=MUTED, style="italic", **FONT)
    save(fig, path)


def build_combined(old_path, new_path, out_path):
    """Stack the old (top) and new (bottom) diagrams into one before/after image."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    top = Image.open(old_path).convert("RGB")
    bot = Image.open(new_path).convert("RGB")
    W = max(top.width, bot.width)
    gap = 150
    canvas = Image.new("RGB", (W, top.height + gap + bot.height), "white")
    canvas.paste(top, ((W - top.width) // 2, 0))
    canvas.paste(bot, ((W - bot.width) // 2, top.height + gap))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            font_manager.findfont("DejaVu Sans:weight=bold"), 34)
    except Exception:
        font = ImageFont.load_default()

    # transition band: a centred [↓ arrow] + caption group
    gy = top.height + gap // 2
    caption = "3 specialists  →  1 vision-language model"
    tb = draw.textbbox((0, 0), caption, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    arrow_w, spacing = 40, 34
    group_w = arrow_w + spacing + tw
    x0 = W // 2 - group_w // 2
    ax_c = x0 + arrow_w // 2            # arrow centre x
    draw.line([(ax_c, gy - 28), (ax_c, gy + 12)], fill="#56616B", width=5)
    draw.polygon([(ax_c - 16, gy + 6), (ax_c + 16, gy + 6), (ax_c, gy + 30)],
                 fill="#56616B")
    draw.text((x0 + arrow_w + spacing, gy - th // 2 - tb[1]), caption,
              fill="#23303B", font=font)

    canvas.save(out_path, "JPEG", quality=92)
    print(f"wrote {out_path}")


def build_outcome(path):
    """Project-outcome diagram: research bet -> negative result -> pivot -> working
    system, then a fork into two paper framings (systems paper = stronger)."""
    fig, ax = plt.subplots(figsize=(16, 11.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.0)
    ax.axis("off")

    NEG = dict(fc="#FDE4E4", ec="#C0392B")   # negative-result stage
    OK = dict(fc="#DBEEDD", ec="#3F9A52")    # success / system

    def panel(cx, cy, w, h, title, lines, *, fc, ec, title_color=INK,
              title_size=11, line_size=8.4, align="center", badge=None,
              badge_fc="#3F9A52", lw=2.0, line_step=0.44):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=lw, facecolor=fc, edgecolor=ec, zorder=2))
        ty = cy + h / 2 - 0.42
        if align == "left":
            tx, ha = cx - w / 2 + 0.38, "left"
        else:
            tx, ha = cx, "center"
        ax.text(tx, ty, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=title_color, **FONT)
        if badge:
            ax.text(cx + w / 2 - 1.1, cy + h / 2 + 0.32, badge, ha="center",
                    va="center", fontsize=9, fontweight="bold", color="white",
                    **FONT, bbox=dict(boxstyle="round,pad=0.36", fc=badge_fc, ec="none"))
        by = ty - 0.62
        for ln in lines:
            ax.text(tx, by, ln, ha=ha, va="center", fontsize=line_size,
                    color=MUTED, **FONT)
            by -= line_step

    ax.text(8.0, 10.5, "Project outcome: from research idea to working system",
            ha="center", va="center", fontsize=18, fontweight="bold", color=INK, **FONT)

    # --- the journey (4 stages, left -> right) ---
    jy, jh, w_s = 8.1, 2.3, 3.5
    cx = [2.05, 6.0, 9.95, 13.9]
    stages = [
        ("The question",
         ["Can meta-learning teach the", "reader a new professor's hand",
          "from a few samples?"], SLIDE, INK),
        ("Reptile",
         ["We built it from scratch.", "No gain on real handwriting:",
          "33% error, level with base."], NEG, RED),
        ("The pivot",
         ["A general vision model,", "untrained, beat the whole",
          "custom pipeline."], DETECT, INK),
        ("The system",
         ["Detect, read text and math,", "typeset back in place.",
          "Deployed. Runs offline."], OK, GREEN_TX),
    ]
    for i, (title, lines, style, tcol) in enumerate(stages, 1):
        panel(cx[i - 1], jy, w_s, jh, title, lines, fc=style["fc"], ec=style["ec"],
              title_color=tcol, title_size=11.5, line_size=8.6, line_step=0.48)
        ax.text(cx[i - 1] - w_s / 2 + 0.5, jy + jh / 2 - 0.45, str(i),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", **FONT,
                bbox=dict(boxstyle="circle,pad=0.30", fc=style["ec"], ec="none"))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        arrow(ax, cx[a] + w_s / 2, jy, cx[b] - w_s / 2, jy)

    ax.text(8.0, 5.95, "How we'd write it up", ha="center", va="center",
            fontsize=11.5, color=INK, fontweight="bold", **FONT,
            bbox=dict(boxstyle="round,pad=0.5", fc="#EDF2F7", ec="#C7D0D8"))

    # --- fork into two paper framings ---
    py, ph = 2.95, 4.0
    rx, sx, rw, sw = 4.3, 11.7, 7.0, 7.4
    panel(rx, py, rw, ph, "Research paper · meta-learning",
          ["• No gain on the target writer at this scale",
           "• Lessons: broad pre-training beats",
           "   specialisation; a good checkpoint beats a",
           "   newer architecture; a small adapter beats",
           "   meta-learning",
           "• One writer, little data, no new method",
           "• Ceiling: a workshop paper"],
          fc=DETECT["fc"], ec=DETECT["ec"], title_color="#B5851F", title_size=12.5,
          line_size=9.0, align="left", badge="NARROWER",
          badge_fc="#D99B2B", lw=2.0, line_step=0.43)
    panel(sx, py, sw, ph, "Systems / application paper",
          ["• First system to combine handwritten text,",
           "   handwritten math, region detection, and",
           "   typeset render-back in one pipeline",
           "• Runs end-to-end on 579 slides; typesets",
           "   ~95% of the math it detects; works offline",
           "• Contribution: the working, evaluated system"],
          fc=OK["fc"], ec=OK["ec"], title_color=GREEN_TX, title_size=12.5,
          line_size=9.0, align="left", badge="★ STRONGER",
          badge_fc="#3F9A52", lw=2.8, line_step=0.43)

    arrow(ax, cx[1], jy - jh / 2, rx + 1.6, py + ph / 2, color="#B5851F", lw=2.2)
    arrow(ax, cx[3], jy - jh / 2, sx - 1.2, py + ph / 2, color=GREEN_TX, lw=2.6)

    ax.text(8.0, 0.5,
            "The research question gave a negative result. The system we built instead "
            "is the stronger paper.",
            ha="center", va="center", fontsize=11, color=MUTED, style="italic", **FONT)
    save(fig, path)


def build_storyboard(outcome_path, arch_path, out_path):
    """Closing 'story' board: the project-outcome diagram stacked on top of the
    before/after architecture, as one tall slide."""
    from PIL import Image, ImageDraw

    top = Image.open(outcome_path).convert("RGB")
    bot = Image.open(arch_path).convert("RGB")
    target_w = min(top.width, bot.width)

    def fit(im):
        if im.width == target_w:
            return im
        h = round(im.height * target_w / im.width)
        return im.resize((target_w, h), Image.LANCZOS)

    top, bot = fit(top), fit(bot)
    gap = 80
    canvas = Image.new("RGB", (target_w, top.height + gap + bot.height), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bot, (0, top.height + gap))

    # subtle hairline divider between the two halves
    draw = ImageDraw.Draw(canvas)
    y = top.height + gap // 2
    m = int(target_w * 0.08)
    draw.line([(m, y), (target_w - m, y)], fill="#C7D0D8", width=3)

    canvas.save(out_path, "JPEG", quality=92)
    print(f"wrote {out_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    figs = root / "figures"
    build_old(figs / "architecture_old.jpg")
    build_new(figs / "architecture_new.jpg")
    build_combined(figs / "architecture_old.jpg",
                   figs / "architecture_new.jpg",
                   figs / "architecture_combined.jpg")
    build_outcome(figs / "project_outcome.jpg")
    build_storyboard(figs / "project_outcome.jpg",
                     figs / "architecture_combined.jpg",
                     figs / "story_board.jpg")


if __name__ == "__main__":
    main()
