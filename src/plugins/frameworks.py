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

from src.app.events import Domain, EdgeID, GovernanceEvent


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
    Example: convert THM displacement signals into edge updates.
    Payload example:
      {
        "domain": "education",
        "GTD": +0.1,
        "IVD": -0.2,
        "IAD": +0.05,
        "IID": +0.0
      }

    This is NOT claiming these are the correct mappings; it's a simple, explicit connector.
    """
    name = "thm_displacement"

    _domain_map = {
        "economy": Domain.ECONOMY,
        "employment": Domain.EMPLOYMENT,
        "education": Domain.EDUCATION,
    }

    def emit_events(self, payload: Dict[str, Any], ctx: PluginContext) -> List[GovernanceEvent]:
        dom = self._domain_map.get(str(payload.get("domain", "")).lower(), None)
        if dom is None:
            return []

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
                    
                    meta_dict = {"plugin": self.name, "signal": key}
                    if ctx.meta:
                        meta_dict.update(ctx.meta)
                    events.append(
                        GovernanceEvent(
                            domain=dom,
                            edge_id=edge,
                            magnitude=val,
                            confidence=confidence,
                            meta=meta_dict,
                        )
                    )
        return events


class GyroscopeWorkMixPlugin(FrameworkPlugin):
    """
    Example: convert Gyroscope work-mix shifts into edge updates.
    Payload example:
      {
        "domain": "employment",
        "GM": +0.1, "ICu": -0.1, "IInter": 0.0, "ICo": 0.0
      }

    Again: explicit, editable policy mapping.
    """
    name = "gyroscope_workmix"

    def emit_events(self, payload: Dict[str, Any], ctx: PluginContext) -> List[GovernanceEvent]:
        dom_str = str(payload.get("domain", "employment")).lower()
        _domain_map = {
            "economy": Domain.ECONOMY,
            "employment": Domain.EMPLOYMENT,
            "education": Domain.EDUCATION,
        }
        dom = _domain_map.get(dom_str, None)
        if dom is None:
            return []

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
            events.append(
                GovernanceEvent(
                    domain=dom,
                    edge_id=EdgeID.GOV_INFO,
                    magnitude=delta_gm_vs_icu,
                    confidence=confidence,
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
            events.append(
                GovernanceEvent(
                    domain=dom,
                    edge_id=EdgeID.INFER_INTEL,
                    magnitude=delta_iinter_vs_ico,
                    confidence=confidence,
                    meta=meta_dict,
                )
            )

        return events

