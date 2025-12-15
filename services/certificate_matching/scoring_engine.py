"""Scoring engine implementing the certificate matching rubric."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from database import CertificateDB
from .profile_builder import RequirementMetric, TenderRequirementProfile
from .utils import normalize_location, parse_inr_amount, tokenize, within_lookback


DEFAULT_WEIGHTS = {
    "sector": 25,
    "services": 20,
    "financial": 20,
    "timeline": 15,
    "geography": 10,
    "metrics": 10,
}


@dataclass
class MatchVerdict:
    certificate_id: str
    score: float
    breakdown: Dict[str, Dict[str, float]]
    matching_factors: List[str]
    gaps: List[str]

    def to_dict(self, certificate: CertificateDB) -> Dict[str, object]:
        return {
            "certificate_id": certificate.id,
            "project_name": certificate.project_name,
            "client_name": certificate.client_name,
            "location": certificate.location,
            "score": self.score,
            "breakdown": self.breakdown,
            "matching_factors": self.matching_factors,
            "gaps": self.gaps,
        }


class CertificateMatchScorer:
    """Score certificates against a compiled tender requirement profile."""

    def __init__(self, profile: TenderRequirementProfile, weights: Optional[Dict[str, float]] = None):
        self.profile = profile
        self.weights = weights or DEFAULT_WEIGHTS
        self.total_available = sum(self.weights.values())

    def score_certificate(self, certificate: CertificateDB) -> MatchVerdict:
        breakdown: Dict[str, Dict[str, float]] = {}
        matching_factors: List[str] = []
        gaps: List[str] = []

        sector_score = self._score_sector(certificate, matching_factors, gaps)
        breakdown['sector'] = self._format_block(sector_score, 'sector')

        service_score = self._score_services(certificate, matching_factors, gaps)
        breakdown['services'] = self._format_block(service_score, 'services')

        financial_score = self._score_financials(certificate, matching_factors, gaps)
        breakdown['financial'] = self._format_block(financial_score, 'financial')

        timeline_score = self._score_timeline(certificate, matching_factors, gaps)
        breakdown['timeline'] = self._format_block(timeline_score, 'timeline')

        geo_score = self._score_geography(certificate, matching_factors, gaps)
        breakdown['geography'] = self._format_block(geo_score, 'geography')

        metric_score = self._score_metrics(certificate, matching_factors, gaps)
        breakdown['metrics'] = self._format_block(metric_score, 'metrics')

        total_score = sum(block['score'] for block in breakdown.values())
        normalized = round((total_score / self.total_available) * 100, 1) if self.total_available else 0.0

        return MatchVerdict(
            certificate_id=certificate.id,
            score=normalized,
            breakdown=breakdown,
            matching_factors=matching_factors,
            gaps=gaps,
        )

    # --- component scoring helpers -------------------------------------------------

    def _format_block(self, score: float, key: str) -> Dict[str, float]:
        return {
            "score": round(score, 2),
            "max_score": float(self.weights.get(key, 0)),
        }

    def _score_sector(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('sector', 0)
        if weight == 0 or not self.profile.sectors:
            return 0.0

        certificate_sectors = tokenize(certificate.sectors) | tokenize(certificate.sub_sectors)
        if not certificate_sectors:
            gaps.append("Certificate lacks sector tagging")
            return 0.0

        overlap = len(certificate_sectors & self.profile.sectors)
        ratio = overlap / max(len(self.profile.sectors), 1)
        if overlap:
            matches.append(f"Sector overlap: {overlap} / {len(self.profile.sectors)}")
        else:
            gaps.append("No overlapping sectors with tender requirement")
        return weight * ratio

    def _score_services(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('services', 0)
        if weight == 0 or not self.profile.services:
            return 0.0

        cert_services = tokenize(certificate.services_rendered)
        if not cert_services:
            gaps.append("Services rendered not captured in certificate")
            return 0.0

        overlap = len(cert_services & self.profile.services)
        ratio = overlap / max(len(self.profile.services), 1)
        if overlap:
            matches.append(f"Services matched: {overlap} / {len(self.profile.services)}")
        else:
            gaps.append("No service overlap")
        return weight * ratio

    def _score_financials(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('financial', 0)
        if weight == 0:
            return 0.0

        min_value = self.profile.min_project_value or 0
        if min_value <= 0:
            return weight  # no requirement, full credit

        certificate_value = certificate.project_value or parse_inr_amount(certificate.project_value_inr)
        if not certificate_value:
            gaps.append("Certificate missing project value")
            return 0.0

        ratio = min(certificate_value / min_value, 1.2)
        if ratio >= 1:
            matches.append(f"Project value meets requirement (₹{certificate_value:,.0f})")
        else:
            gaps.append("Project value below tender threshold")
        return weight * (ratio / 1.2)

    def _score_timeline(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('timeline', 0)
        if weight == 0:
            return 0.0

        lookback = max(self.profile.years_lookback, 1)
        completion_date = certificate.end_date or certificate.completion_date
        if within_lookback(completion_date, lookback):
            matches.append(f"Completed within last {lookback} years")
            return float(weight)

        gaps.append("Certificate is older than lookback period")
        if completion_date:
            years_old = max(0, datetime.utcnow().year - completion_date.year)
            decay = max(0.0, 1 - (years_old - lookback) / lookback)
            return weight * max(decay, 0.0)
        return 0.0

    def _score_geography(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('geography', 0)
        if weight == 0 or not self.profile.locations:
            return 0.0

        certificate_location = normalize_location(certificate.location)
        if not certificate_location:
            gaps.append("Location not captured")
            return 0.0

        if certificate_location in self.profile.locations:
            matches.append("Geography requirement satisfied")
            return float(weight)

        gaps.append("Location differs from tender preference")
        return weight * 0.25  # partial credit for any location present

    def _score_metrics(self, certificate: CertificateDB, matches: List[str], gaps: List[str]) -> float:
        weight = self.weights.get('metrics', 0)
        if weight == 0 or not self.profile.metrics:
            return 0.0

        certificate_metrics = certificate.metrics or []
        if not certificate_metrics:
            gaps.append("Metrics missing in certificate")
            return 0.0

        metric_hits = 0
        for requirement in self.profile.metrics:
            if self._certificate_has_metric(requirement, certificate_metrics):
                metric_hits += 1

        if metric_hits:
            matches.append(f"Metrics matched: {metric_hits} / {len(self.profile.metrics)}")
        else:
            gaps.append("No numerical metrics matched")

        ratio = metric_hits / max(len(self.profile.metrics), 1)
        return weight * ratio

    def _certificate_has_metric(self, requirement: RequirementMetric, certificate_metrics: List[Dict[str, object]]) -> bool:
        required_name = requirement.name.lower()
        for metric in certificate_metrics:
            metric_name = str(metric.get('metric_name') or metric.get('name') or '').lower()
            if not metric_name:
                continue
            if required_name and required_name not in metric_name:
                continue
            if requirement.min_value is None:
                return True

            value = metric.get('value')
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if numeric_value >= float(requirement.min_value):
                return True

        return False

