# Layer 3: Static Notice Router
# [Authority:Indirect] + [Agency:Indirect]

import re
from typing import Optional


class StaticRouter:
    """Routes notices when displacement expressions are detected."""

    # Static notice - never generated, always this exact text
    NOTICE = (
        "This text is classified as [Authority:Indirect]. "
        "All decisions based on this classification remain with [Agency:Direct]."
    )

    def _validate_expression(self, expression: str) -> bool:
        """
        Validate that expression conforms to the canonical THM PEG grammar.
        Implements lexer and parser for the grammar in THM_Grammar.md.
        """
        if not expression or not expression.strip():
            return False

        try:
            return self._parse_thm_expression(expression.strip())
        except:
            return False

    def _parse_thm_expression(self, expr: str) -> bool:
        """
        Parse THM expression according to the PEG grammar:
        expression   <- displacement / flow / tag
        displacement <- tag ">" tag "=" risk
        flow         <- tag "->" tag
        tag          <- composite / simple / negated
        composite    <- simple ("+" simple)+
        simple       <- "[" category ":" value "]" / "[" concept "]"
        negated      <- "!" simple
        category     <- "Authority" / "Agency"
        value        <- "Direct" / "Indirect"
        concept      <- "Information" / "Inference" / "Intelligence"
        risk         <- "[Risk:" risk_code "]"
        risk_code    <- "GTD" / "IVD" / "IAD" / "IID"
        """
        expr = expr.strip()

        # Try to parse as displacement
        if self._parse_displacement(expr):
            return True

        # Try to parse as flow
        if self._parse_flow(expr):
            return True

        # Try to parse as single tag
        if self._parse_tag(expr):
            return True

        return False

    def _parse_displacement(self, expr: str) -> bool:
        """Parse displacement: tag ">" tag "=" risk"""
        # Look for the pattern: ... > ... = [Risk:...]
        if '=' not in expr or '>' not in expr:
            return False

        eq_pos = expr.find('=')
        gt_pos = expr.find('>')

        if eq_pos < gt_pos:
            return False  # = before > is invalid

        # Split into parts
        left_part = expr[:gt_pos].strip()
        middle_part = expr[gt_pos+1:eq_pos].strip()
        right_part = expr[eq_pos+1:].strip()

        # Validate parts
        if not self._parse_tag(left_part):
            return False
        if not self._parse_tag(middle_part):
            return False
        if not self._parse_risk(right_part):
            return False

        return True

    def _parse_flow(self, expr: str) -> bool:
        """Parse flow: tag "->" tag (can be chained)"""
        if '->' not in expr:
            return False

        # Split by -> and validate each tag
        parts = expr.split('->')
        if len(parts) < 2:
            return False

        for part in parts:
            if not self._parse_tag(part.strip()):
                return False

        return True

    def _parse_tag(self, tag: str) -> bool:
        """Parse tag: composite / simple / negated"""
        tag = tag.strip()

        # Try negated
        if tag.startswith('!'):
            return self._parse_simple(tag[1:].strip())

        # Try composite (contains +)
        if '+' in tag:
            subtags = tag.split('+')
            if len(subtags) < 2:
                return False
            for subtag in subtags:
                if not self._parse_simple(subtag.strip()):
                    return False
            return True

        # Try simple
        return self._parse_simple(tag)

    def _parse_simple(self, tag: str) -> bool:
        """Parse simple tag: "[" category ":" value "]" / "[" concept "]\""""
        tag = tag.strip()

        if not (tag.startswith('[') and tag.endswith(']')):
            return False

        content = tag[1:-1].strip()

        # Check if it's category:value format
        if ':' in content:
            parts = content.split(':', 1)
            if len(parts) != 2:
                return False
            category, value = parts[0].strip(), parts[1].strip()

            # Validate category and value
            if category not in ['Authority', 'Agency']:
                return False
            if value not in ['Direct', 'Indirect']:
                return False
            return True

        # Check if it's concept format
        else:
            if content not in ['Information', 'Inference', 'Intelligence']:
                return False
            return True

    def _parse_risk(self, risk: str) -> bool:
        """Parse risk: "[Risk:" risk_code "]\""""
        risk = risk.strip()

        if not (risk.startswith('[Risk:') and risk.endswith(']')):
            return False

        risk_code = risk[6:-1].strip()
        if risk_code not in ['GTD', 'IVD', 'IAD', 'IID']:
            return False

        return True

    def is_valid(self, expression: str) -> bool:
        """
        Public method to validate THM grammar expressions.
        """
        return self._validate_expression(expression)

    def route(self, expression: str) -> Optional[str]:
        """
        Route expression to notice if displacement is detected.

        Args:
            expression: THM grammar expression string

        Returns:
            Static notice string if displacement detected, None otherwise
        """
        if not self._validate_expression(expression):
            return None

        # Check for displacement operator '>' and risk tag
        has_displacement = '>' in expression
        has_risk = '[Risk:' in expression

        if has_displacement and has_risk:
            return self.NOTICE

        return None


# Global router instance
router = StaticRouter()