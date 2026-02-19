# Gyroscope 2.0 Trace Builder
# [Authority:Indirect] + [Agency:Indirect]
#
# Deterministic template. No model generates any part of this.

from datetime import datetime, timezone

from .context import CONSULT_ALIGNED, CONSULT_DISPLACEMENT, THM_GRAMMAR, THM_MARK

_FIRST_TRACE_TEMPLATE = """[Gyroscope 2.0]

{mark}

{grammar}

1 = Governance Management Traceability
2 = Information Curation Variety
3 = Inference Interaction Accountability
4 = Intelligence Cooperation Integrity
[THM: {thm_expression}]
[Consult: {consult}]
[Timestamp: {timestamp} | ID: {trace_id:03d}]
[End]"""

_TRACE_TEMPLATE = """[Gyroscope 2.0]
1 = Governance Management Traceability
2 = Information Curation Variety
3 = Inference Interaction Accountability
4 = Intelligence Cooperation Integrity
[THM: {thm_expression}]
[Consult: {consult}]
[Timestamp: {timestamp} | ID: {trace_id:03d}]
[End]"""


def build_trace(
    thm_expression: str,
    risk_code: str | None,
    trace_id: int,
    timestamp: str | None = None,
    first: bool = False,
) -> str:
    """Build a Gyroscope 2.0 trace block with THM classification.

    Args:
        thm_expression: Well-formed THM grammar expression from GyroGem.
        risk_code: Extracted risk code or None.
        trace_id: Sequential trace identifier.
        timestamp: ISO timestamp (YYYY-MM-DDTHH:MM). Generated if not provided.
        first: If True, include full Mark + Grammar in the trace.

    Returns:
        Complete Gyroscope 2.0 trace block string.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    if risk_code and risk_code in CONSULT_DISPLACEMENT:
        consult = CONSULT_DISPLACEMENT[risk_code]
    else:
        consult = CONSULT_ALIGNED

    if first:
        return _FIRST_TRACE_TEMPLATE.format(
            mark=THM_MARK.strip(),
            grammar=THM_GRAMMAR.strip(),
            thm_expression=thm_expression,
            consult=consult,
            timestamp=timestamp,
            trace_id=trace_id,
        )
    else:
        return _TRACE_TEMPLATE.format(
            thm_expression=thm_expression,
            consult=consult,
            timestamp=timestamp,
            trace_id=trace_id,
        )
