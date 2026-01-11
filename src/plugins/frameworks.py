"""
Internal connectors: CGM / Gyroscope / THM.

Goal:
- Convert domain-specific internal signals (employment categories, THM displacement tags, etc.)
  into GovernanceEvents that update the App ledgers.
- No semantics in the kernel. Plugins live here.

This file provides:
- a minimal plugin interface
- a few conservative, explicit mapping helpers

NOTE:
Mapping from "a real-world assessment" -> "which edge changes" is an application policy.
We keep it explicit and auditable, not hidden in math tricks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.app.events import Domain, EdgeID, GovernanceEvent, MICRO


@dataclass(frozen=True)
class PluginContext:
    """
    Minimal context plugins might use.
    """
    actor_id: Optional[str] = None
    session_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class FrameworkPlugin:
    """
    Base class for internal framework plugins.
    """
    name: str = "framework_plugin"

    def emit_events(self, payload: Dict[str, Any], ctx: PluginContext) -> List[GovernanceEvent]:
        """
        Convert a payload into 0..N GovernanceEvents.
        Must be deterministic for a given payload.
        """
        raise NotImplementedError


# -------------------------
# Example connectors (minimal)
# -------------------------

class THMDisplacementPlugin(FrameworkPlugin):
    """
    Convert THM displacement signals into edge updates.
    
    THM displacements (GTD, IVD, IAD, IID) are measurements of risk signatures.
    Per GGG hierarchy: THM = Education domain (measurements/displacements).
    
    All events from this plugin are emitted to EDUCATION domain regardless of payload.
    
    Payload example:
      {
        "GTD": +0.1,
        "IVD": -0.2,
        "IAD": +0.05,
        "IID": +0.0
      }
    """
    name = "thm_displacement"

    def emit_events(self, payload: Dict[str, Any], ctx: PluginContext) -> List[GovernanceEvent]:
        # THM displacements always go to Education domain (measurements level)
        dom = Domain.EDUCATION

        # Minimal, explicit edge choices:
        # - GTD affects Governance–Information coupling
        # - IVD affects Information–Inference coupling
        # - IAD affects Governance–Inference coupling
        # - IID affects Inference–Intelligence coupling
        # These are policy choices; keep them visible and editable.
        mapping = [
            ("GTD", EdgeID.GOV_INFO),
            ("IVD", EdgeID.INFO_INFER),
            ("IAD", EdgeID.GOV_INFER),
            ("IID", EdgeID.INFER_INTEL),
        ]

        events: List[GovernanceEvent] = []
        for key, edge in mapping:
            if key in payload:
                val = float(payload[key])
                if val != 0.0:
                    # Use per-signal confidence if available, fallback to global confidence
                    signal_confidence_key = f"{key}_confidence"
                    confidence = float(payload.get(signal_confidence_key, payload.get("confidence", 1.0)))
                    
                    # Convert to micro-units
                    magnitude_micro = int(round(val * MICRO))
                    confidence_micro = int(round(confidence * MICRO))
                    
                    meta_dict = {"plugin": self.name, "signal": key}
                    if ctx.meta:
                        meta_dict.update(ctx.meta)
                    events.append(
                        GovernanceEvent(
                            domain=dom,
                            edge_id=edge,
                            magnitude_micro=magnitude_micro,
                            confidence_micro=confidence_micro,
                            meta=meta_dict,
                        )
                    )
        return events


class GyroscopeWorkMixPlugin(FrameworkPlugin):
    """
    Convert Gyroscope work-mix shifts into edge updates.
    
    Gyroscope alignment work (GMT, ICV, IIA, ICI) represents active principles.
    Per GGG hierarchy: Gyroscope = Employment domain (active work/principles).
    
    All events from this plugin are emitted to EMPLOYMENT domain regardless of payload.
    
    Payload example:
      {
        "GM": +0.1, "ICu": -0.1, "IInter": 0.0, "ICo": 0.0
      }
    """
    name = "gyroscope_workmix"

    def emit_events(self, payload: Dict[str, Any], ctx: PluginContext) -> List[GovernanceEvent]:
        # Gyroscope alignment work always goes to Employment domain (work level)
        dom = Domain.EMPLOYMENT

        gm = float(payload.get("GM", 0.0))
        icu = float(payload.get("ICu", 0.0))
        iinter = float(payload.get("IInter", 0.0))
        ico = float(payload.get("ICo", 0.0))

        events: List[GovernanceEvent] = []

        # Example: if governance management rises relative to information curation,
        # increase Gov–Info tension.
        delta_gm_vs_icu = gm - icu
        if delta_gm_vs_icu != 0.0:
            # Use minimum confidence of the two signals involved
            gm_conf = float(payload.get("GM_confidence", payload.get("confidence", 1.0)))
            icu_conf = float(payload.get("ICu_confidence", payload.get("confidence", 1.0)))
            confidence = min(gm_conf, icu_conf)
            
            meta_dict = {"plugin": self.name, "metric": "GM-ICu"}
            if ctx.meta:
                meta_dict.update(ctx.meta)
            # Convert to micro-units
            magnitude_micro = int(round(delta_gm_vs_icu * MICRO))
            confidence_micro = int(round(confidence * MICRO))
            
            events.append(
                GovernanceEvent(
                    domain=dom,
                    edge_id=EdgeID.GOV_INFO,
                    magnitude_micro=magnitude_micro,
                    confidence_micro=confidence_micro,
                    meta=meta_dict,
                )
            )

        # If inference interaction shifts relative to intelligence cooperation,
        # affect Infer–Intel coupling.
        delta_iinter_vs_ico = iinter - ico
        if delta_iinter_vs_ico != 0.0:
            # Use minimum confidence of the two signals involved
            iinter_conf = float(payload.get("IInter_confidence", payload.get("confidence", 1.0)))
            ico_conf = float(payload.get("ICo_confidence", payload.get("confidence", 1.0)))
            confidence = min(iinter_conf, ico_conf)
            
            meta_dict = {"plugin": self.name, "metric": "IInter-ICo"}
            if ctx.meta:
                meta_dict.update(ctx.meta)
            # Convert to micro-units
            magnitude_micro = int(round(delta_iinter_vs_ico * MICRO))
            confidence_micro = int(round(confidence * MICRO))
            
            events.append(
                GovernanceEvent(
                    domain=dom,
                    edge_id=EdgeID.INFER_INTEL,
                    magnitude_micro=magnitude_micro,
                    confidence_micro=confidence_micro,
                    meta=meta_dict,
                )
            )

        return events

