"""Build tender requirement profiles from stored tender + criteria data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from database import TenderDB, TenderEligibilityCriteriaDB
from .utils import infer_years_from_text, normalize_location, parse_inr_amount, tokenize


DEFAULT_SCORE_THRESHOLD = 50.0


@dataclass
class RequirementMetric:
    name: str
    min_value: Optional[float] = None
    unit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str | float]]:
        return {
            "metric_name": self.name,
            "min_value": self.min_value,
            "unit": self.unit,
            "notes": self.notes,
        }


@dataclass
class TenderRequirementProfile:
    tender_id: str
    tender_title: str
    tender_number: Optional[str]
    tender_state: Optional[str]
    sectors: Set[str] = field(default_factory=set)
    subsectors: Set[str] = field(default_factory=set)
    services: Set[str] = field(default_factory=set)
    min_project_value: Optional[float] = None
    min_consultancy_fee: Optional[float] = None
    similar_works_count: Optional[int] = None
    years_lookback: int = 7
    locations: Set[str] = field(default_factory=set)
    metrics: List[RequirementMetric] = field(default_factory=list)
    funding_agencies: Set[str] = field(default_factory=set)
    score_threshold: float = DEFAULT_SCORE_THRESHOLD

    def to_dict(self) -> Dict[str, object]:
        return {
            "tender_id": self.tender_id,
            "tender_title": self.tender_title,
            "tender_number": self.tender_number,
            "tender_state": self.tender_state,
            "required_sectors": sorted(self.sectors),
            "required_subsectors": sorted(self.subsectors),
            "required_services": sorted(self.services),
            "min_project_value": self.min_project_value,
            "min_consultancy_fee": self.min_consultancy_fee,
            "similar_works_count": self.similar_works_count,
            "years_lookback": self.years_lookback,
            "location_requirement": sorted(self.locations),
            "technical_metrics": [metric.to_dict() for metric in self.metrics],
            "funding_agencies": sorted(self.funding_agencies),
            "score_threshold": self.score_threshold,
        }


class TenderRequirementBuilder:
    """Aggregate structured requirements for a tender matching run."""

    def __init__(self, tender: TenderDB, criteria: Iterable[TenderEligibilityCriteriaDB]):
        self.tender = tender
        self.criteria = list(criteria)

    def build(self) -> TenderRequirementProfile:
        profile = TenderRequirementProfile(
            tender_id=self.tender.id,
            tender_title=self.tender.title or (self.tender.work_item_details or {}).get('Title', '') or 'Tender',
            tender_number=self.tender.tender_reference_number,
            tender_state=self.tender.state,
        )

        self._bootstrap_from_tender(profile)
        self._ingest_criteria(profile)

        if not profile.locations and self.tender.state:
            normalized = normalize_location(self.tender.state)
            if normalized:
                profile.locations.add(normalized)

        if profile.min_project_value is None and isinstance(self.tender.estimated_value, (int, float)):
            profile.min_project_value = float(self.tender.estimated_value) * 0.4  # Typical "single work" rule

        return profile

    def _bootstrap_from_tender(self, profile: TenderRequirementProfile) -> None:
        work_details = self.tender.work_item_details or {}
        fee_details = self.tender.tender_fee_details or {}

        profile.sectors.update(tokenize(self.tender.category))
        profile.subsectors.update(tokenize(work_details.get('Sub Sector')))
        profile.services.update(tokenize(work_details.get('Work Description')))

        estimated_value = work_details.get('Tender Value (INR)') if isinstance(work_details, dict) else None
        if estimated_value:
            parsed_value = parse_inr_amount(estimated_value)
            if parsed_value:
                profile.min_project_value = max(profile.min_project_value or 0, parsed_value)

        if isinstance(fee_details, dict):
            fee_value = fee_details.get('EMD Amount') or fee_details.get('Document Fee')
            fee_numeric = parse_inr_amount(fee_value)
            if fee_numeric:
                profile.min_consultancy_fee = max(profile.min_consultancy_fee or 0, fee_numeric)

    def _ingest_criteria(self, profile: TenderRequirementProfile) -> None:
        for criterion in self.criteria:
            requirements = criterion.extracted_requirements or {}
            keywords = criterion.keywords or []
            tokens = tokenize(keywords) or tokenize(criterion.criteria_text)

            if criterion.category in {"scope_of_work", "services", "technical"}:
                profile.services.update(tokens)
            elif criterion.category in {"authority", "experience"}:
                profile.sectors.update(tokens)
                profile.subsectors.update(tokens)

            if criterion.category == "metrics" and requirements:
                metric_name = requirements.get('metric_name') or criterion.criteria_text[:80]
                profile.metrics.append(
                    RequirementMetric(
                        name=metric_name,
                        min_value=requirements.get('min_value') or requirements.get('value'),
                        unit=requirements.get('unit'),
                        notes=requirements.get('notes') or None,
                    )
                )

            if criterion.category == "financial" and requirements.get('min_value') is not None:
                min_value = requirements.get('min_value')
                unit = requirements.get('unit')
                parsed_financial = parse_inr_amount(f"{min_value} {unit}" if unit else min_value)
                if parsed_financial:
                    profile.min_project_value = max(profile.min_project_value or 0, parsed_financial)

            timeframe = requirements.get('timeframe_years')
            if timeframe is None:
                inferred = infer_years_from_text(criterion.criteria_text)
                if inferred:
                    timeframe = inferred
            if timeframe:
                profile.years_lookback = max(profile.years_lookback, int(timeframe))

            location = requirements.get('location') or requirements.get('geography')
            if location:
                normalized_location = normalize_location(location)
                if normalized_location:
                    profile.locations.add(normalized_location)

            if criterion.category == "experience" and requirements.get('similar_works_count'):
                profile.similar_works_count = max(profile.similar_works_count or 0, int(requirements['similar_works_count']))

            funding = requirements.get('funding_agency')
            if funding:
                profile.funding_agencies.add(funding.lower())
