"""Minimal Markdown -> LaTeX conversion for paper export (headings, emphasis, lists, code, math, links)."""
from __future__ import annotations

import re

_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "\\": r"\textbackslash{}", "$": r"\$"}


def _esc(s: str) -> str:
    return "".join(_SPECIAL.get(c, c) for c in s)


def _inline(text: str) -> str:
    # Protect math and code spans, escape the rest, then restore.
    tokens: list[str] = []

    def keep(s):
        tokens.append(s)
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(f"\\[{m.group(1)}\\]"), text)
    text = re.sub(r"\$(.+?)\$", lambda m: keep(f"${m.group(1)}$"), text)
    text = re.sub(r"`([^`]+)`", lambda m: keep(r"\texttt{" + _esc(m.group(1)) + "}"), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: keep(r"\href{" + m.group(2) + "}{" + _esc(m.group(1)) + "}"),
                  text)
    text = re.sub(r"\b((?:run|f|exp)-\d{8}-\d{6}-[0-9a-f]{4})\b", lambda m: keep(r"\texttt{" + _esc(m.group(1)) + "}"),
                  text)
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], text)


def markdown_to_latex(md: str, title: str) -> str:
    out = [r"\documentclass{article}", r"\usepackage{amsmath,amssymb,hyperref,graphicx}",
           r"\title{" + _esc(title) + "}", r"\date{\today}", r"\begin{document}", r"\maketitle", ""]
    lines = md.splitlines()
    i = 0
    in_list = None
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            j = i + 1
            block = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            out += [r"\begin{verbatim}", *block, r"\end{verbatim}"]
            i = j + 1
            continue
        if line.strip() == "$$":
            j = i + 1
            block = []
            while j < len(lines) and lines[j].strip() != "$$":
                block.append(lines[j])
                j += 1
            out += [r"\begin{equation*}", *block, r"\end{equation*}"]
            i = j + 1
            continue
        m_list = re.match(r"^\s*([-*]|\d+\.)\s+(.*)", line)
        if m_list:
            kind = "enumerate" if m_list.group(1)[0].isdigit() else "itemize"
            if in_list != kind:
                if in_list:
                    out.append(rf"\end{{{in_list}}}")
                out.append(rf"\begin{{{kind}}}")
                in_list = kind
            out.append(r"\item " + _inline(m_list.group(2)))
            i += 1
            continue
        if in_list:
            out.append(rf"\end{{{in_list}}}")
            in_list = None
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            if level == 1:
                i += 1
                continue  # the title is \maketitle
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
            out.append(rf"\{cmd}{{{_inline(m.group(2))}}}")
        elif line.startswith(">"):
            out.append(r"\begin{quote}" + _inline(line.lstrip("> ")) + r"\end{quote}")
        else:
            out.append(_inline(line))
        i += 1
    if in_list:
        out.append(rf"\end{{{in_list}}}")
    out += ["", r"\end{document}", ""]
    return "\n".join(out)
