# Layer 1: Regex Gate
# [Authority:Indirect] + [Agency:Indirect]

import re
from typing import List, Tuple, Optional


class RegexGate:
    """Determines whether text contains grammatical markers for source-type confusion."""

    def __init__(self):
        # Define regex patterns for different source types
        self.patterns = {
            'system_prompt': self._system_prompt_patterns(),
            'model_output': self._model_output_patterns(),
            'user_input': self._user_input_patterns()
        }

    def _system_prompt_patterns(self) -> List[str]:
        """Patterns for system prompts - self-presentation as [Agency:Direct]"""
        return [
            # Copula constructions: bounded to avoid spanning large blocks
            r'\b(?:I\s+am|we\s+are|you\s+are|it\s+is)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Modal constructions: bounded to sentence-like segments
            r'\b(?:I\s+will|we\s+will|you\s+will|it\s+will|I\s+can|we\s+can|you\s+can|it\s+can|I\s+should|we\s+should|you\s+should|it\s+should|I\s+may|we\s+may|you\s+may|it\s+may|I\s+might|we\s+might|you\s+might|it\s+might|I\s+could|we\s+could|you\s+could|it\s+could|I\s+must|we\s+must|you\s+must|it\s+must)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with copula
            r'\b(?:I\'m|we\'re|you\'re|it\'s)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with modals
            r'\b(?:I\'ll|we\'ll|you\'ll|it\'ll|I\'d|we\'d|you\'d|it\'d)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b'
        ]

    def _model_output_patterns(self) -> List[str]:
        """Patterns for model output - attributing [Agency:Direct] capacities to self"""
        return [
            # Copula: bounded to avoid spanning large blocks
            r'\b(?:I\s+am|we\s+are)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Modal: bounded to sentence-like segments
            r'\b(?:I\s+will|we\s+will|I\s+can|we\s+can|I\s+should|we\s+should|I\s+may|we\s+may|I\s+might|we\s+might|I\s+could|we\s+could|I\s+must|we\s+must)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with copula
            r'\b(?:I\'m|we\'re)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with modals
            r'\b(?:I\'ll|we\'ll|I\'d|we\'d)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b'
        ]

    def _user_input_patterns(self) -> List[str]:
        """Patterns for user input - assigning [Agency:Direct] role to [Agency:Indirect]"""
        return [
            # Copula: bounded to avoid spanning large blocks
            r'\b(?:you\s+are|it\s+is|the\s+AI\s+is|this\s+system\s+is)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Modal: bounded to sentence-like segments
            r'\b(?:you\s+will|it\s+will|you\s+can|it\s+can|you\s+should|it\s+should|you\s+may|it\s+may|you\s+might|it\s+might|you\s+could|it\s+could|you\s+must|it\s+must|the\s+AI\s+will|this\s+system\s+will)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with copula
            r'\b(?:you\'re|it\'s)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Contractions with modals
            r'\b(?:you\'ll|it\'ll|you\'d|it\'d)\s+(?:\w+\s+){0,5}\w+[\.!?]?\b',
            # Role assignment patterns (act as, be a, pretend to be, etc.)
            r'\b(?:act\s+as|be\s+a|be\s+an|pretend\s+to\s+be|role\s+play\s+as|you\s+are\s+(?:a|an))\s+(?:\w+\s+){0,5}\w+[\.!?]?\b'
        ]

    def check(self, text: str, source: str) -> Tuple[bool, List[Tuple[int, int, str]]]:
        """
        Check if text contains displacement triggers.

        Args:
            text: Text span to check
            source: One of 'system_prompt', 'model_output', 'user_input'

        Returns:
            Tuple of (triggered: bool, matched_spans: List[Tuple[start, end, pattern]])
        """
        if source not in self.patterns:
            raise ValueError(f"Unknown source type: {source}")

        matched_spans = []
        for pattern in self.patterns[source]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_spans.append((match.start(), match.end(), pattern))

        return len(matched_spans) > 0, matched_spans


# Global gate instance
gate = RegexGate()