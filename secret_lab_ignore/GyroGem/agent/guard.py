# GyroGem Guard - Orchestrator
# [Authority:Indirect] + [Agency:Indirect]


from .context import DEFAULT_EXPRESSION
from .model import GyroGemModel
from .router import THMRouter
from .trace import build_trace


class GyroGemGuard:
    """Orchestrates GyroGem classification and Gyroscope trace construction.

    Runs every turn. No gate. No filtering.
    First invocation produces a trace carrying the full Mark + Grammar.
    Subsequent invocations produce compact traces.
    """

    def __init__(self, model: GyroGemModel | None = None,
                 router: THMRouter | None = None):
        self.model = model or GyroGemModel()
        self.router = router or THMRouter()
        self._trace_counter = 0
        self._first_trace_sent = False

    def process(self, text: str) -> dict:
        """Classify assistant output and build Gyroscope trace.

        Args:
            text: Full assistant message.

        Returns:
            Dict with:
                expression: THM grammar expression
                risk_code: str or None
                is_displacement: bool
                trace: Complete Gyroscope 2.0 trace block
                first_trace: bool
        """
        self._trace_counter += 1

        # Classify
        try:
            expression = self.model.classify(text)
        except Exception:
            expression = None

        # Validate and extract risk
        if expression and self.router.validate(expression):
            risk_code = self.router.extract_risk(expression)
        else:
            expression = DEFAULT_EXPRESSION
            risk_code = None

        # Build trace
        is_first = not self._first_trace_sent
        trace = build_trace(
            thm_expression=expression,
            risk_code=risk_code,
            trace_id=self._trace_counter,
            first=is_first,
        )
        self._first_trace_sent = True

        return {
            "expression": expression,
            "risk_code": risk_code,
            "is_displacement": risk_code is not None,
            "trace": trace,
            "first_trace": is_first,
        }
