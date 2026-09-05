"""Excerpt extraction with a verbatim-substring guarantee.

Audit note: an earlier version joined the best *non-consecutive* lines
with spaces, so excerpts were paraphrases rather than quotes — most of
them failed a strict ``excerpt in corpus[source]`` check. Every helper
here returns a contiguous slice of the original text, so the check
always passes.
"""

from __future__ import annotations

from config import EXCERPT_CONTEXT_LINES, EXCERPT_LIMIT


def _line_hits(line: str, terms: list[str]) -> int:
    lowered = line.lower()
    return sum(lowered.count(t) for t in terms if t)


def best_excerpt(text: str, terms: list[str],
                 limit: int = EXCERPT_LIMIT, context: int | None = None) -> str:
    """Return a compact contiguous quote rich in the requested terms.

    The window is a *contiguous* slice of original lines (joined with
    ``"\\n"``), grown around the strongest hit until ``limit`` is
    reached, so the result is always a verbatim ``in``-substring.
    Pass ``context=0`` for row-oriented files (CSV) where neighbouring
    lines are unrelated records.
    """
    if context is None:
        context = EXCERPT_CONTEXT_LINES
    if not text.strip():
        return ""
    wanted = [t.lower() for t in terms if t]
    raw_lines = text.splitlines()
    # Skip blank padding lines but remember original indices so the
    # final join stays contiguous.
    idx = [i for i, ln in enumerate(raw_lines) if ln.strip()]
    if not idx:
        return text[:limit].strip()
    if not wanted:
        start = idx[0]
        return "\n".join(raw_lines[start:start + 3])[:limit].strip()

    scored = [
        (_line_hits(raw_lines[i], wanted), i) for i in idx
    ]
    scored = [(h, i) for h, i in scored if h > 0]
    if not scored:
        start = idx[0]
        return "\n".join(raw_lines[start:start + 3])[:limit].strip()

    scored.sort(key=lambda p: (-p[0], p[1]))
    best = scored[0][1]
    # Minimal contiguous span covering the top two hits when it fits —
    # this keeps e.g. an exception line *and* its matching failure line.
    span_lo = span_hi = best
    if len(scored) > 1:
        second = scored[1][1]
        lo, hi = min(best, second), max(best, second)
        candidate = "\n".join(raw_lines[lo:hi + 1])
        if len(candidate) <= limit:
            span_lo, span_hi = lo, hi
    # Grow the contiguous window around the span until the budget is
    # reached, so short hits (e.g. a runbook heading) still pull in
    # their remediation/MTTR lines below. Never grow across a markdown
    # section header: quoting RB-014 must not bleed RB-002's (different,
    # unconfirmed) MTTR into the same excerpt.
    pos = idx.index(span_lo)
    end_pos = idx.index(span_hi)
    lo_pos = max(0, pos - max(context, 0))
    hi_pos = min(len(idx) - 1, end_pos + max(context, 0))

    def _is_header(i: int) -> bool:
        return raw_lines[i].lstrip().startswith("#")

    lo_open, hi_open = True, True  # False once we hit a section boundary
    while True:
        grown = False
        if lo_pos > 0 and lo_open:
            cand = "\n".join(
                raw_lines[idx[lo_pos - 1]:idx[hi_pos] + 1]
            ).strip()
            if len(cand) <= limit:
                lo_pos -= 1
                grown = True
                # A header may join the quote, but the quote must not
                # reach past it into the previous section.
                lo_open = not _is_header(idx[lo_pos])
        if hi_pos < len(idx) - 1 and hi_open:
            if _is_header(idx[hi_pos + 1]):
                hi_open = False  # next section starts here — stop
            else:
                cand = "\n".join(
                    raw_lines[idx[lo_pos]:idx[hi_pos + 1] + 1]
                ).strip()
                if len(cand) <= limit:
                    hi_pos += 1
                    grown = True
        if not grown:
            break
    span_lo, span_hi = idx[lo_pos], idx[hi_pos]
    excerpt = "\n".join(raw_lines[span_lo:span_hi + 1]).strip()
    if len(excerpt) > limit:
        # Prefer a line boundary, then a word boundary; either way the
        # result stays a verbatim prefix of the contiguous slice.
        # (A "line cut" only counts when a newline actually exists.)
        cut = excerpt[:limit]
        if "\n" in cut:
            line_cut = cut.rsplit("\n", 1)[0].rstrip()
            if len(line_cut) >= limit // 2:
                excerpt = line_cut
            else:
                excerpt = (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip()
        else:
            excerpt = (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip()
    # Safety net: never return a non-verbatim quote.
    if excerpt not in text:
        excerpt = text.strip()[:limit].strip()
    return excerpt
