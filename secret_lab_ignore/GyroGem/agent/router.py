# THM Expression Validation and Risk Extraction
# [Authority:Indirect] + [Agency:Indirect]

import re
from typing import Optional


_RISK_RE = re.compile(r"\[Risk:(GTD|IVD|IAD|IID)\]")
_CAT_TAG_RE = re.compile(r"^\[(Authority|Agency)\s*:\s*(Direct|Indirect)\]$")
_CONCEPT_TAG_RE = re.compile(r"^\[(Information|Inference|Intelligence)\]$")


class THMRouter:
    """Validates THM grammar expressions and extracts risk codes.

    This validator is whitespace-tolerant and accepts expressions with or without
    spaces around operators, consistent with the THM PEG grammar.
    """

    def validate(self, expression: str) -> bool:
        if not expression or not expression.strip():
            return False
        try:
            return self._parse_expression(expression.strip())
        except Exception:
            return False

    def extract_risk(self, expression: str) -> Optional[str]:
        expr = (expression or "").strip()
        if not self.validate(expr):
            return None
        if not self._contains_displacement(expr):
            return None
        m = _RISK_RE.search(expr)
        return m.group(1) if m else None

    def is_displacement(self, expression: str) -> bool:
        return self.extract_risk(expression) is not None

    def _parse_expression(self, expr: str) -> bool:
        return self._parse_displacement(expr) or self._parse_flow(expr) or self._parse_tag(expr)

    def _contains_displacement(self, expr: str) -> bool:
        # ">" that is not part of "->"
        return re.search(r"(?<!-)>", expr) is not None

    def _parse_displacement(self, expr: str) -> bool:
        # displacement <- tag ">" tag "=" risk
        gt_positions = [m.start() for m in re.finditer(r"(?<!-)>", expr)]
        if not gt_positions:
            return False
        gt_pos = gt_positions[0]

        eq_pos = expr.find("=", gt_pos + 1)
        if eq_pos < 0:
            return False

        left = expr[:gt_pos].strip()
        middle = expr[gt_pos + 1 : eq_pos].strip()
        right = expr[eq_pos + 1 :].strip()

        return self._parse_tag(left) and self._parse_tag(middle) and self._parse_risk(right)

    def _parse_flow(self, expr: str) -> bool:
        # flow <- tag ("->" tag)+
        if "->" not in expr:
            return False
        parts = re.split(r"\s*->\s*", expr)
        if len(parts) < 2:
            return False
        return all(self._parse_tag(p.strip()) for p in parts)

    def _parse_tag(self, tag: str) -> bool:
        # tag <- composite / simple / negated
        tag = tag.strip()
        if not tag:
            return False

        if tag.startswith("!"):
            inner = tag[1:].strip()
            return self._parse_simple(inner)

        if "+" in tag:
            subtags = re.split(r"\s*\+\s*", tag)
            if len(subtags) < 2:
                return False
            return all(self._parse_simple(s.strip()) for s in subtags)

        return self._parse_simple(tag)

    def _parse_simple(self, tag: str) -> bool:
        # simple <- "[" category ":" value "]" / "[" concept "]"
        tag = tag.strip()
        if _CAT_TAG_RE.match(tag):
            return True
        if _CONCEPT_TAG_RE.match(tag):
            return True
        return False

    def _parse_risk(self, risk: str) -> bool:
        # risk <- "[Risk:" risk_code "]"
        risk = risk.strip()
        return _RISK_RE.fullmatch(risk) is not None