"""LLM-powered certificate-to-tender matching."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI

from .profile_builder import TenderRequirementProfile
from .scoring_engine import MatchVerdict

logger = logging.getLogger(__name__)

AI_MATCH_MODEL = os.getenv("TENDER_AI_MATCH_MODEL", "gpt-4o-mini")

MATCH_PROMPT_TEMPLATE = """You are an infrastructure tender analyst. Evaluate whether the completion certificate satisfies the tender eligibility requirements.

Return ONLY valid JSON in the following structure:
{{
  "score": 0-100 (whole number),
  "matching_factors": ["..."],
  "gaps": ["..."],
  "summary": "One sentence conclusion"
}}

Scoring guidance:
- >=90: Certificate clearly exceeds every key requirement.
- 75-89: Meets most requirements with minor gaps.
- 60-74: Partial alignment; notable gaps exist.
- <60: Weak match.

Consider sectors, services, project value, metrics, geography, timeline, JV role, funding agencies.

Tender Requirements JSON:
{tender_json}

Certificate JSON:
{certificate_json}
"""


@dataclass
class CertificateAIMatcher:
    """LLM-powered matcher that scores a certificate against a tender profile."""

    client: OpenAI

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not configured for AI certificate matching")
        self.client = OpenAI(api_key=api_key)

    def _serialize_profile(self, profile: TenderRequirementProfile) -> Dict[str, Any]:
        data = profile.to_dict()
        return {
            "tender_title": data.get("tender_title"),
            "tender_number": data.get("tender_number"),
            "sectors": data.get("required_sectors"),
            "services": data.get("required_services"),
            "min_project_value": data.get("min_project_value"),
            "min_consultancy_fee": data.get("min_consultancy_fee"),
            "similar_works_count": data.get("similar_works_count"),
            "years_lookback": data.get("years_lookback"),
            "locations": data.get("location_requirement"),
            "technical_metrics": data.get("technical_metrics"),
            "funding_agencies": data.get("funding_agencies"),
        }

    def _serialize_certificate(self, certificate) -> Dict[str, Any]:
        return {
            "project_name": certificate.project_name,
            "client_name": certificate.client_name,
            "sectors": certificate.sectors,
            "sub_sectors": certificate.sub_sectors,
            "services_rendered": certificate.services_rendered,
            "scope_of_work": certificate.scope_of_work,
            "project_value": certificate.project_value,
            "consultancy_fee_inr": certificate.consultancy_fee_inr,
            "location": certificate.location,
            "metrics": certificate.metrics,
            "role_lead_jv": certificate.role_lead_jv,
            "jv_partners": certificate.jv_partners,
            "funding_agency": certificate.funding_agency,
            "completion_date": certificate.end_date or certificate.completion_date,
        }

    def score_certificate(self, profile: TenderRequirementProfile, certificate) -> MatchVerdict:
        tender_json = json.dumps(self._serialize_profile(profile), ensure_ascii=False)
        certificate_json = json.dumps(self._serialize_certificate(certificate), ensure_ascii=False)

        prompt = MATCH_PROMPT_TEMPLATE.format(
            tender_json=tender_json,
            certificate_json=certificate_json,
        )

        response = self.client.chat.completions.create(
            model=AI_MATCH_MODEL,
            temperature=0.2,
            max_completion_tokens=800,
            messages=[
                {"role": "system", "content": "You are a concise infrastructure tender analyst."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("AI matcher returned non-JSON response: %s", content)
            parsed = {
                "score": 0,
                "matching_factors": ["AI response not parseable"],
                "gaps": [],
                "summary": "AI response was invalid JSON.",
            }

        score = parsed.get("score") or 0
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

        matching_factors = parsed.get("matching_factors") or []
        if isinstance(matching_factors, str):
            matching_factors = [matching_factors]

        gaps = parsed.get("gaps") or []
        if isinstance(gaps, str):
            gaps = [gaps]

        summary = parsed.get("summary") or ""
        matching_factors = [str(item) for item in matching_factors]
        gaps = [str(item) for item in gaps]

        breakdown = {
            "ai_score": {"score": round(score, 2), "max_score": 100.0},
        }

        return MatchVerdict(
            certificate_id=certificate.id,
            score=round(score, 1),
            breakdown=breakdown,
            matching_factors=matching_factors or [summary or "AI approved"],
            gaps=gaps,
        )
