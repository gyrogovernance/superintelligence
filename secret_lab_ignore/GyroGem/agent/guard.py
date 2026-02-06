# GyroGem Guard - Orchestrator
# [Authority:Indirect] + [Agency:Indirect]

import hashlib
from typing import Dict, Tuple, List, Optional
from .gate import gate
from .model import model
from .router import router


class GyroGemGuard:
    """Orchestrates the three-layer GyroGem pipeline."""

    def __init__(self):
        self.system_prompt_cache: Dict[str, Dict] = {}

    def _cache_key(self, text: str) -> str:
        """Generate cache key for system prompt."""
        return hashlib.sha256(text.encode()).hexdigest()

    def process(self, text: str, source: str) -> Dict:
        """
        Process text through the GyroGem pipeline.

        Args:
            text: Text span to classify
            source: One of 'system_prompt', 'model_output', 'user_input'

        Returns:
            Dict with keys: triggered, expression, notice, matched_spans
        """
        # Handle system prompt caching
        if source == 'system_prompt':
            cache_key = self._cache_key(text)
            if cache_key in self.system_prompt_cache:
                return self.system_prompt_cache[cache_key].copy()

        # Layer 1: Regex Gate
        triggered, matched_spans = gate.check(text, source)

        if not triggered:
            result = {
                'triggered': False,
                'expression': None,
                'notice': None,
                'matched_spans': matched_spans
            }
        else:
            # Extract bounded span for model input (token economy)
            model_span = self._extract_model_span(text, matched_spans)

            # Layer 2: Model Classification
            expression = model.classify(model_span, source)

            # Layer 3: Router
            notice = router.route(expression)

            result = {
                'triggered': True,
                'expression': expression,
                'notice': notice,
                'matched_spans': matched_spans
            }

        # Cache system prompt results
        if source == 'system_prompt':
            cache_key = self._cache_key(text)
            self.system_prompt_cache[cache_key] = result.copy()

        return result

    def _extract_model_span(self, text: str, matched_spans: List[Tuple[int, int, str]]) -> str:
        """
        Extract a bounded span from matched spans for model input.
        Implements token economy by not passing full text.
        """
        if not matched_spans:
            return text  # Fallback to full text if no spans

        # Choose the first span (simplest strategy)
        start, end, pattern = matched_spans[0]

        # Expand to sentence boundaries or bounded window
        # Simple approach: expand to include context but stay bounded
        window_start = max(0, start - 100)  # 100 chars before
        window_end = min(len(text), end + 100)  # 100 chars after

        span_text = text[window_start:window_end]

        # Ensure we have at least the matched portion
        if len(span_text.strip()) < 10:  # Too short, expand
            span_text = text[max(0, start-200):min(len(text), end+200)]

        return span_text.strip()


# Global guard instance
guard = GyroGemGuard()