#!/usr/bin/env python3
"""Flag the things the style guide forbids. Usage: python scripts/lint_prose.py docs/index.html

Checks paragraph, list and caption text (not code blocks, tables or the charts). It is deliberately strict: a flag is a prompt to
look, and some are fine in context. The lists-of-three rule cannot be checked reliably by a script, so read for it yourself.
"""
import re, sys

BANNED_WORDS = ["journey", "delve", "landscape", "leverage", "seamless", "robust", "game-changer", "crucial", "testament"]
src = open(sys.argv[1] if len(sys.argv) > 1 else "docs/index.html", encoding="utf-8").read()
src = re.sub(r"<!--.*?-->|<style.*?</style>|<script.*?</script>|<svg.*?</svg>|<pre.*?</pre>|<table.*?</table>", "", src, flags=re.S)
body = src[src.find("<main"):]
issues = []

for tag in ("strong", "b", "em", "i"):
    if re.search(rf"<{tag}[ >]", body): issues.append(f"emphasis tag <{tag}> used")
if re.search(r"<ul", body): issues.append("a bullet list is present: is prose better?")

blocks = re.findall(r"<(?:p|li|figcaption)[^>]*>(.*?)</(?:p|li|figcaption)>", body, flags=re.S)
text = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", b)).strip() for b in blocks]
words = 0
for t in text:
    words += len(t.split())
    for s in [x for x in re.split(r"(?<=[.!?])\s+", t) if x]:
        n = len(s.split())
        if "\u2014" in s or "\u2013" in s: issues.append(f"dash: {s[:70]}")
        if "(" in s: issues.append(f"parenthesis: {s[:70]}")
        if "!" in s: issues.append(f"exclamation: {s[:70]}")
        if "?" in s: issues.append(f"question: {s[:70]}")
        if ":" in s: issues.append(f"colon (a reveal?): {s[:70]}")
        if n <= 5: issues.append(f"very short sentence ({n} words): {s}")
        if n > 45: issues.append(f"very long sentence ({n} words): {s[:70]}...")
        if re.search(r"\b(?:is|was|are|were|isn't|wasn't|not)\b[^.]{0,40}\bnot\b[^.]{0,25},? (?:it's|it is|but)\b", s, re.I) or re.search(r"\bnot just\b", s, re.I):
            issues.append(f"'not X, it's Y' shape: {s[:70]}")
        for w in BANNED_WORDS:
            if re.search(rf"\b{w}", s, re.I): issues.append(f"banned word '{w}': {s[:70]}")
        if re.search(r"\b(here's|here is) (the|what)\b", s, re.I): issues.append(f"colon-reveal phrasing: {s[:70]}")

print(f"{words} words of prose")
print("\n".join(sorted(set(issues))) if issues else "no flags")
sys.exit(1 if issues else 0)
