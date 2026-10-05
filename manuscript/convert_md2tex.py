#!/usr/bin/env python3
"""Convert manuscript_v1.md to TMLR LaTeX. Run: python3 convert_md2tex.py [--preprint|--anonymous]"""
import re, sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "--preprint"
SRC = "/home/hatch/workspace/review-lottery/manuscript/manuscript_v1.md"
OUT = "/home/hatch/workspace/review-lottery/manuscript/latex/review_lottery.tex"

md = open(SRC).read()

# Single-pass literal replacement: longest keys first, one re.sub over alternation
# so replacement VALUES are never re-scanned.
def make_replacer(pairs):
    keys = sorted([k for k, _ in pairs], key=len, reverse=True)
    pat = re.compile("|".join(re.escape(k) for k in keys))
    d = dict(pairs)
    return lambda s: pat.sub(lambda m: d[m.group(0)], s)

MATH_PAIRS = [
    ("sigma^2_paper,y", r"$\sigma^2_{\mathrm{paper},y}$"),
    ("sigma^2_paper", r"$\sigma^2_{\mathrm{paper}}$"),
    ("rho_y", r"$\rho_y$"),
    ("theta_i", r"$\theta_i$"),
    ("s_ij", r"$s_{ij}$"),
    ("mu_y", r"$\mu_y$"),
    ("c_y", r"$c_y$"),
    ("k_i", r"$k_i$"),
    ("alpha_year", r"$\alpha_{\mathrm{year}}$"),
    ("beta*confidence", r"$\beta\cdot\mathrm{confidence}$"),
    ("beta*conf", r"$\beta\cdot\mathrm{conf}$"),
    ("P(flip | accepted)", r"$P(\mathrm{flip}\mid\mathrm{accepted})$"),
    ("P(flip | rejected)", r"$P(\mathrm{flip}\mid\mathrm{rejected})$"),
    ("P(accept | mean_score, sd_score, year)",
     r"$P(\mathrm{accept}\mid\mathrm{mean\_score},\mathrm{sd\_score},\mathrm{year})$"),
    ("P(accept)", r"$P(\mathrm{accept})$"),
    ("rho", r"$\rho$"),
    ("k=2/3/4", r"$k=2/3/4$"),
    ("k=2", r"$k=2$"), ("k=3", r"$k=3$"), ("k=4", r"$k=4$"),
    ("B=1,000", r"$B=1{,}000$"),
    ("{1,3,6,8}", r"$\{1,3,6,8\}$"),
    ("{1,3,5,6,8,10}", r"$\{1,3,5,6,8,10\}$"),
    ("1-10 scale", r"$1$--$10$ scale"),
    ("rho ~ year", r"$\rho \sim \mathrm{year}$"),
    ("R^2", r"$R^2$"),
    ("+/-", r"$\pm$"),
    ("n=9", r"$n=9$"),
    ("n=166", r"$n=166$"), ("n=882", r"$n=882$"),
    ("n=28,476", r"$n=28{,}476$"), ("n=35,617", r"$n=35{,}617$"),
    ("n=18,740", r"$n=18{,}740$"),
    ("43/166 = 25.9%", r"$43/166 = 25.9\%$"),
    ("203/882 = 23.0%", r"$203/882 = 23.0\%$"),
    ("2+2", r"$2{+}2$"),
    ("3.37 * mean_score - 0.14 * sd_score",
     r"$3.37\cdot\mathrm{mean\_score}-0.14\cdot\mathrm{sd\_score}$"),
]
math_repl = make_replacer(MATH_PAIRS)

CITE_PAIRS = [
    ("(Cortes and Lawrence, arXiv:2109.09774)", r"\citep{cortes2021}"),
    ("(Beygelzimer et al., arXiv:2306.03262)", r"\citep{beygelzimer2023}"),
    ("Tran et al. (arXiv:2010.05137)", r"Tran et al.~\citep{tran2020}"),
    ("Liang et al. (arXiv:2403.07183)", r"Liang et al.~\citep{liang2024}"),
    ("(arXiv:2109.09774)", r"~\citep{cortes2021}"),
    ("(arXiv:2306.03262)", r"~\citep{beygelzimer2023}"),
    ("(arXiv:2010.05137)", r"~\citep{tran2020}"),
    ("(arXiv:2403.07183)", r"~\citep{liang2024}"),
    ("(arXiv:2509.25701)", r"~\citep{acceptance2025}"),
]
cite_repl = make_replacer(CITE_PAIRS)

PP_PAIRS = [
    ("+0.3pp", r"+0.3\,pp"), ("-2.6pp", r"$-2.6$\,pp"),
    ("+5pp", r"+5\,pp"), ("+0.04", r"$+0.04$"),
    ("+7pp", r"+7\,pp"), ("+11pp", r"+11\,pp"),
    ("-5.4pp", r"$-5.4$\,pp"), ("-3.2pp", r"$-3.2$\,pp"),
    ("-3 to -5pp", r"$-3$ to $-5$\,pp"),
    ("3-5pp", r"$3$--$5$\,pp"), ("1pp", r"1\,pp"),
    ("24-fold", r"$24$-fold"), ("5-7%", r"$5$--$7$\%"),
    ("2017-2019 and 2021", "2017--2019 and 2021"),
    ("2017-2025", "2017--2025"), ("2018-2022", "2018--2022"),
    ("2022-2025", "2022--2025"), ("2021-2025", "2021--2025"),
    ("2023-2025", "2023--2025"), ("2017-2020", "2017--2020"),
]
pp_repl = make_replacer(PP_PAIRS)

def inline_fmt(s):
    s = re.sub(r"`([^`]+)`",
               lambda m: r"\texttt{" + m.group(1).replace("_", r"\_").replace("%", r"\%") + "}",
               s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    return s


def esc_text_outside_math(s):
    """Escape & % # _ outside $...$ math segments (lookbehind protects existing backslash-escapes)."""
    parts = re.split(r'(\$.*?\$)', s)
    for i in range(0, len(parts), 2):
        q = parts[i]
        q = re.sub(r'(?<!\\)&', r'\\&', q)
        q = re.sub(r'(?<!\\)%', r'\\%', q)
        q = re.sub(r'(?<!\\)#', r'\\#', q)
        q = re.sub(r'(?<!\\)_', r'\\_', q)
        parts[i] = q
    return ''.join(parts)

def md_table_to_latex(lines, caption, label, colspec=None, small=False):
    rows = []
    for ln in lines:
        ln = ln.strip().strip("|")
        rows.append([c.strip() for c in ln.split("|")])
    data = [r for r in rows if not all(re.match(r"^:?-{2,}:?$", c) for c in r)]
    ncol = len(data[0])
    if colspec is None:
        colspec = "l" + "r" * (ncol - 1)
    out = [r"\begin{table}[t]", r"\centering"]
    if small:
        out.append(r"\small")
    out.append(r"\caption{" + caption + r"}\label{" + label + "}")
    out.append(r"\begin{tabular}{" + colspec + r"}\toprule")
    for i, r in enumerate(data):
        cells = []
        for c in r:
            c = inline_fmt(c)
            c = cite_repl(c); c = math_repl(c); c = pp_repl(c)
            c = esc_text_outside_math(c)
            cells.append(c)
        out.append(" & ".join(cells) + r" \\")
        if i == 0:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)

lines = md.split("\n")
title = "The Review Lottery: Calibrating an Observational Estimator of Peer-Review Noise (ICLR 2017-2025)"
ab_start = next(i for i, l in enumerate(lines) if l.strip() == "## Abstract")
ab_end = next(i for i, l in enumerate(lines) if l.strip().startswith("## 1."))
abstract = " ".join(l.strip() for l in lines[ab_start+1:ab_end] if l.strip())
abstract = cite_repl(abstract); abstract = math_repl(abstract); abstract = pp_repl(abstract)
abstract = inline_fmt(abstract)
abstract = esc_text_outside_math(abstract)
body_lines = lines[ab_end:]
ref_idx = next((i for i, l in enumerate(body_lines) if l.strip() == "## References"), None)
if ref_idx is not None:
    body_lines = body_lines[:ref_idx]

FIGS = [
    ("As a sensitivity check, a pure threshold rule",
     "fig_decision_calibration.png",
     r"Calibration of the logistic decision model: predicted versus observed acceptance rate (5-fold CV AUC $=0.933$). The model tracks the ideal diagonal closely.",
     "fig:decision-calib", "0.62\\textwidth"),
    ("Table 1. Signal-to-noise",
     "fig_rho.png",
     r"Signal-to-noise ratio $\rho_y$ by year (posterior mean, 95\% credible interval). The dashed line marks the NeurIPS 2014 estimate of about 50\% subjective variance.",
     "fig:rho", "0.85\\textwidth"),
    ("a bias of +0.3 percentage points, essentially unbiased",
     "fig_calib_external.png",
     r"External calibration: simulated two-committee disagreement at $k=2$ and $k=3$ against the NeurIPS 2014 (25.9\%) and 2021 (23.0\%) experimental values. At the matched $k=3$ caliber the 2021 simulation sits on the observed value.",
     "fig:calib-external", "0.85\\textwidth"),
    ("Answer to the preregistered checkpoint",
     "fig_calib_internal.png",
     r"Internal calibration: $k=2$ model simulation versus the model-free $2{+}2$ split-half anchor (200 repetitions). Agreement is within 1\,pp from 2021 onward.",
     "fig:calib-internal", "0.85\\textwidth"),
    ("the conclusion is that there is no robust, attributable time trend",
     "fig_rho_trend.png",
     r"RQ-B: $\rho$ over 2017--2025 with the OLS trend ($-0.0122$/year, $p=0.110$; no trend). The shaded band marks the 2020 four-point scale.",
     "fig:rho-trend", "0.85\\textwidth"),
    ("Combined with the multivariate null",
     "fig_disagreement_trend.png",
     r"RQ-B: simulated disagreement over 2017--2025 at $k=2$ and $k=3$ with the NeurIPS 2014 and 2021 anchors for reference. The nominal rise is not robust to dropping single years.",
     "fig:disagreement-trend", "0.85\\textwidth"),
    ("2021 shows no compression fingerprint",
     "fig_scale_compression.png",
     r"2020/2021 anomaly descriptives. (a) Normalized score dispersion; (b) mean cutpoint spacing relative to $\sigma_{\mathrm{paper}}$; (c) fraction of papers with zero within-paper variance (23.7\% in 2020); (d) accepted-paper flip probability against $\rho$. The 2020 four-point scale (shaded) compresses scores; 2021 instead shows low signal with threshold crowding.",
     "fig:scale-compression", "0.95\\textwidth"),
    ("the data cannot support a conclusion about LLM effects",
     "fig_llm_descriptives.png",
     r"RQ-C descriptives (no causal claims). (a) Within-paper score variance with a 2023 breakpoint (ITS: level $p=0.864$, slope $p=0.766$; near-zero power); (b) mean review length, PeerSum 2018--2022 only; (c) mean reviewer confidence, PeerSum 2018--2022 only. No post-2022 review text or confidence data exist.",
     "fig:llm-descriptives", "0.95\\textwidth"),
]

tex = []
tex.append(r"\documentclass[10pt]{article}")
tex.append(r"\usepackage[preprint]{tmlr}" if MODE == "--preprint" else r"\usepackage{tmlr}")
tex.append(r"\input{math_commands.tex}")
tex.append(r"\usepackage{hyperref}")
tex.append(r"\usepackage{url}")
tex.append(r"\usepackage{booktabs}")
tex.append(r"\usepackage{graphicx}")
tex.append("")
tex.append(r"\title{" + title + "}")
tex.append("")
if MODE == "--preprint":
    tex.append(r"\author{\name Feilian Huang \\ \addr Independent researcher}")
else:
    tex.append("% anonymous submission: author block hidden by tmlr style")
tex.append("")
tex.append(r"\def\month{10}")
tex.append(r"\def\year{2026}")
tex.append(r"\def\openreview{\url{https://openreview.net/forum?id=XXXX}}")
tex.append("")
tex.append(r"\begin{document}")
tex.append(r"\maketitle")
tex.append("")
tex.append(r"\begin{abstract}")
tex.append(abstract)
tex.append(r"\end{abstract}")
tex.append("")

fig_queue = list(FIGS)

def flush_figs_before(text_chunk, out):
    global fig_queue
    pending, remaining = [], []
    for item in fig_queue:
        (remaining, pending)[item[0] in text_chunk].append(item)
    fig_queue = remaining
    for anchor, fn, cap, lab, w in pending:
        out += [r"\begin{figure}[t]", r"\centering",
                r"\includegraphics[width=" + w + "]{" + fn + "}",
                r"\caption{" + cap + "}", r"\label{" + lab + "}",
                r"\end{figure}"]

i, n = 0, len(body_lines)
while i < n:
    ln = body_lines[i]
    s = ln.strip()
    if not s:
        tex.append(""); i += 1; continue
    if s.startswith("> "):
        tex.append("% " + s[2:]); i += 1; continue
    if s.startswith("### "):
        head = re.sub(r"^\d+\.\d+\s*", "", s[4:])
        tex.append(r"\subsection{" + inline_fmt(head) + "}"); i += 1; continue
    if s.startswith("## "):
        head = re.sub(r"^\d+\.\s*", "", s[3:])
        tex.append(r"\section{" + inline_fmt(head) + "}"); i += 1; continue
    m = re.match(r"Table (\d+)\.\s*(.*?)(?:\s*<!--\s*(.*?)\s*-->)?$", s)
    if m:
        num, cap, src = m.group(1), m.group(2), m.group(3)
        j = i + 1
        while j < n and not body_lines[j].strip():
            j += 1
        if j >= n or not body_lines[j].strip().startswith("|"):
            tex.append("% TABLE CAPTION WITHOUT TABLE: " + s); i += 1; continue
        tlines = []
        while j < n and body_lines[j].strip().startswith("|"):
            tlines.append(body_lines[j]); j += 1
        cap = cite_repl(cap); cap = math_repl(cap); cap = pp_repl(cap)
        cap = inline_fmt(cap)
        cap = esc_text_outside_math(cap)
        flush_figs_before("Table " + num + ". " + m.group(2), tex)
        ttex = md_table_to_latex(tlines, cap, "tab:t" + num)
        if src:
            ttex += "\n% source: " + src
        tex.append(ttex); i = j; continue
    if s.startswith("| # | Claim |"):
        j = i; tlines = []
        while j < n and body_lines[j].strip().startswith("|"):
            tlines.append(body_lines[j]); j += 1
        cap = "Claim-to-evidence checklist: each substantive claim maps to generated evidence."
        tex.append(md_table_to_latex(tlines, cap, "tab:claims",
                                     colspec="c p{5.2cm} p{5.2cm} p{3.2cm}", small=True))
        i = j; continue
    if s.startswith("|"):
        tex.append("% UNHANDLED TABLE LINE: " + s); i += 1; continue
    if s.startswith("<!--") and s.endswith("-->"):
        tex.append("%" + s[4:-3]); i += 1; continue
    m2 = re.match(r"^(\d+)\.\s+(.*)$", s)
    if m2:
        item = m2.group(2)
        item = cite_repl(item); item = math_repl(item); item = pp_repl(item)
        item = inline_fmt(item)
        item = esc_text_outside_math(item)
        if not tex or tex[-1] in ("",) or not tex[-1].startswith("\\item"):
            # open list if previous line wasn't an item
            prev = next((t for t in reversed(tex) if t.strip()), "")
            if not prev.startswith("\\item"):
                tex.append(r"\begin{enumerate}")
        tex.append(r"\item " + item)
        nxt = body_lines[i+1].strip() if i + 1 < n else ""
        if not re.match(r"^\d+\.\s+", nxt):
            tex.append(r"\end{enumerate}")
        i += 1; continue
    para = s
    comments = re.findall(r"<!--(.*?)-->", para)
    para = re.sub(r"\s*<!--.*?-->", "", para)
    para = cite_repl(para); para = math_repl(para); para = pp_repl(para)
    para = inline_fmt(para)
    para = esc_text_outside_math(para)
    flush_figs_before(para, tex)
    tex.append(para)
    for c in comments:
        tex.append("%" + c.strip())
    i += 1

for anchor, fn, cap, lab, w in fig_queue:
    tex.append("% UNANCHORED FIGURE: " + fn)
    tex += [r"\begin{figure}[t]", r"\centering",
            r"\includegraphics[width=" + w + "]{" + fn + "}",
            r"\caption{" + cap + "}", r"\label{" + lab + "}",
            r"\end{figure}"]

tex.append("")
tex.append(r"\begin{thebibliography}{9}")
tex.append(r"\bibitem[Beygelzimer et al.(2023)]{beygelzimer2023}")
tex.append("Alina Beygelzimer, Yann N. Dauphin, Percy Liang, Jennifer Wortman Vaughan. "
           r"Has the Machine Learning Review Process Become More Arbitrary as the Field Has Grown? "
           r"The NeurIPS 2021 Consistency Experiment. \textit{arXiv:2306.03262 [cs.LG]}, 2023. "
           r"\url{https://arxiv.org/abs/2306.03262}")
tex.append("")
tex.append(r"\bibitem[Cortes and Lawrence(2021)]{cortes2021}")
tex.append("Corinna Cortes, Neil D. Lawrence. "
           r"Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment. "
           r"\textit{arXiv:2109.09774 [cs.DL]}, 2021. \url{https://arxiv.org/abs/2109.09774}")
tex.append("")
tex.append(r"\bibitem[Liang et al.(2024)]{liang2024}")
tex.append("Weixin Liang et al. "
           r"Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews. "
           r"\textit{arXiv:2403.07183 [cs.CL]}, 2024. \url{https://arxiv.org/abs/2403.07183}")
tex.append("")
tex.append(r"\bibitem[Tran et al.(2020)]{tran2020}")
tex.append("David Tran et al. "
           r"An Open Review of OpenReview: A Critical Analysis of the Machine Learning Conference Review Process. "
           r"\textit{arXiv:2010.05137 [cs.LG]}, 2020. \url{https://arxiv.org/abs/2010.05137}")
tex.append("")
tex.append(r"\bibitem[Process-centric(2025)]{acceptance2025}")
tex.append("What Drives Paper Acceptance? A Process-Centric Analysis of Modern Peer Review. "
           r"\textit{arXiv:2509.25701}, 2025. \url{https://arxiv.org/abs/2509.25701}")
tex.append(r"\end{thebibliography}")
tex.append("")
tex.append(r"\end{document}")

open(OUT, "w").write("\n".join(tex))
print("wrote", OUT, "unanchored figs:", len(fig_queue))
