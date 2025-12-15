"""Certificate-to-tender matching engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import joinedload

from database import (
    SessionLocal,
    CertificateDB,
    TenderDB,
    TenderEligibilityCriteriaDB,
    TenderMatchDB,
    TenderMatchResultDB,
)
from services.certificate_matching import (
    CertificateMatchScorer,
    MatchVerdict,
    TenderRequirementBuilder,
    TenderRequirementProfile,
    CertificateAIMatcher,
)


logger = logging.getLogger(__name__)


class CertificateTenderMatcher:
    """Primary orchestration class for certificate matching runs."""

    def __init__(self, default_threshold: float = 50.0):
        self.default_threshold = default_threshold
        self.ai_matcher = None

    def match_certificates_to_tender(self, tender_id: str, user_id: str, mode: str = "keyword") -> Dict[str, Any]:
        session = SessionLocal()

        try:
            tender = session.query(TenderDB).filter(TenderDB.id == tender_id).first()
            if not tender:
                raise ValueError("Tender not found")

            criteria = (
                session.query(TenderEligibilityCriteriaDB)
                .filter(TenderEligibilityCriteriaDB.tender_id == tender_id)
                .order_by(TenderEligibilityCriteriaDB.page_number)
                .all()
            )

            profile = TenderRequirementBuilder(tender, criteria).build()
            if profile.score_threshold < self.default_threshold:
                profile.score_threshold = self.default_threshold

            certificates = (
                session.query(CertificateDB)
                .options(joinedload(CertificateDB.batch))
                .filter(
                    CertificateDB.user_id == user_id,
                    CertificateDB.processing_status == 'completed'
                )
                .all()
            )

            mode = self._normalize_mode(mode)
            certificate_lookup = {cert.id: cert for cert in certificates}

            if not certificates:
                logger.info("No certificates available for user %s", user_id)
                return self._empty_response(tender, profile, total_certificates=0, mode=mode)

            processed_certificate_ids: Optional[Set[str]] = None
            if mode == "ai":
                cached_match = self._get_latest_match_for_mode(session, tender_id, user_id, mode)
                processed_certificate_ids = set()
                cached_verdicts = self._hydrate_cached_verdicts(cached_match, certificate_lookup)
                if cached_match:
                    cached_summary = cached_match.summary or {}
                    processed_certificate_ids.update(cached_summary.get("processed_certificates") or [])

                unprocessed_certificates = [
                    cert for cert in certificates
                    if cert.id not in processed_certificate_ids
                ]

                if not unprocessed_certificates and cached_match:
                    logger.info("AI matching cache hit for tender %s / user %s", tender_id, user_id)
                    cached_verdicts.sort(key=lambda item: item[1].score, reverse=True)
                    return self._build_response(
                        tender,
                        profile,
                        certificates,
                        cached_verdicts,
                        cached_match.id,
                        mode=mode,
                    )

                new_verdicts = self._score_with_ai(profile, unprocessed_certificates)
                processed_certificate_ids.update(cert.id for cert in unprocessed_certificates)
                verdicts = cached_verdicts + new_verdicts
            else:
                verdicts = self._score_with_keyword(profile, certificates)

            if processed_certificate_ids is None:
                processed_certificate_ids_list = None
            else:
                processed_certificate_ids_list = sorted(processed_certificate_ids)

            verdicts.sort(key=lambda item: item[1].score, reverse=True)

            match_record = self._persist_match(
                session,
                tender,
                user_id,
                profile,
                certificates,
                verdicts,
                mode=mode,
                processed_certificate_ids=processed_certificate_ids_list,
            )

            response = self._build_response(
                tender,
                profile,
                certificates,
                verdicts,
                match_record.id,
                mode=mode,
            )
            session.commit()
            return response
        except Exception:
            session.rollback()
            logger.exception("Certificate matching failed for tender %s", tender_id)
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------

    def _score_with_keyword(
        self,
        profile: TenderRequirementProfile,
        certificates: List[CertificateDB],
    ) -> List[Tuple[CertificateDB, MatchVerdict]]:
        scorer = CertificateMatchScorer(profile)
        verdicts: List[Tuple[CertificateDB, MatchVerdict]] = []
        for certificate in certificates:
            verdict = scorer.score_certificate(certificate)
            if verdict.score >= profile.score_threshold:
                verdicts.append((certificate, verdict))
        return verdicts

    def _score_with_ai(
        self,
        profile: TenderRequirementProfile,
        certificates: List[CertificateDB],
    ) -> List[Tuple[CertificateDB, MatchVerdict]]:
        if self.ai_matcher is None:
            try:
                self.ai_matcher = CertificateAIMatcher()
            except Exception as exc:
                logger.error("Failed to initialize AI matcher: %s", exc)
                return []

        verdicts: List[Tuple[CertificateDB, MatchVerdict]] = []
        for certificate in certificates:
            try:
                verdict = self.ai_matcher.score_certificate(profile, certificate)
                if verdict.score >= profile.score_threshold:
                    verdicts.append((certificate, verdict))
            except Exception as exc:
                logger.warning("AI matching failed for certificate %s: %s", certificate.id, exc)
                continue
        return verdicts

    def _get_latest_match_for_mode(
        self,
        session,
        tender_id: str,
        user_id: str,
        mode: str,
    ) -> Optional[TenderMatchDB]:
        matches = (
            session.query(TenderMatchDB)
            .options(
                joinedload(TenderMatchDB.results).joinedload(TenderMatchResultDB.certificate)
            )
            .filter(
                TenderMatchDB.tender_id == tender_id,
                TenderMatchDB.user_id == user_id,
            )
            .order_by(TenderMatchDB.created_at.desc())
        )

        for match in matches:
            summary = match.summary or {}
            if summary.get("matching_method") == mode:
                return match
        return None

    def _hydrate_cached_verdicts(
        self,
        cached_match: Optional[TenderMatchDB],
        certificate_lookup: Dict[str, CertificateDB],
    ) -> List[Tuple[CertificateDB, MatchVerdict]]:
        if not cached_match:
            return []

        verdicts: List[Tuple[CertificateDB, MatchVerdict]] = []
        for result in cached_match.results or []:
            certificate = certificate_lookup.get(result.certificate_id) or result.certificate
            if not certificate:
                continue
            verdicts.append((
                certificate,
                MatchVerdict(
                    certificate_id=result.certificate_id,
                    score=result.score,
                    breakdown=result.breakdown or {},
                    matching_factors=result.matching_factors or [],
                    gaps=result.gaps or [],
                ),
            ))
        return verdicts

    def _persist_match(
        self,
        session,
        tender: TenderDB,
        user_id: str,
        profile,
        certificates: List[CertificateDB],
        verdicts: List[Tuple[CertificateDB, MatchVerdict]],
        mode: str,
        processed_certificate_ids: Optional[List[str]] = None,
    ) -> TenderMatchDB:
        summary = {
            "total_certificates": len(certificates),
            "matched_certificates": len(verdicts),
            "best_score": verdicts[0][1].score if verdicts else 0,
            "profile": profile.to_dict(),
            "matching_method": mode,
        }
        if processed_certificate_ids is not None:
            summary["processed_certificates"] = processed_certificate_ids

        match_record = TenderMatchDB(
            tender_id=tender.id,
            user_id=user_id,
            tender_name=tender.title or "Tender",
            tender_number=tender.tender_reference_number,
            client_authority=tender.authority,
            required_sectors=list(profile.sectors),
            required_subsectors=list(profile.subsectors),
            required_services=list(profile.services),
            min_project_value=profile.min_project_value,
            min_consultancy_fee=profile.min_consultancy_fee,
            similar_works_count=profile.similar_works_count,
            years_lookback=profile.years_lookback,
            location_requirement=list(profile.locations),
            technical_metrics=[metric.to_dict() for metric in profile.metrics],
            funding_agencies=list(profile.funding_agencies),
            score_threshold=profile.score_threshold,
            summary=summary,
        )

        session.add(match_record)
        session.flush()

        for certificate, verdict in verdicts:
            session.add(
                TenderMatchResultDB(
                    match_id=match_record.id,
                    certificate_id=certificate.id,
                    score=verdict.score,
                    breakdown=verdict.breakdown,
                    matching_factors=verdict.matching_factors,
                    gaps=verdict.gaps,
                )
            )

        return match_record

    def _build_response(
        self,
        tender: TenderDB,
        profile,
        certificates: List[CertificateDB],
        verdicts: List[Tuple[CertificateDB, MatchVerdict]],
        match_id: str,
        mode: str,
    ) -> Dict[str, Any]:
        return {
            "match_id": match_id,
            "tender_id": tender.id,
            "tender_title": tender.title,
            "profile": profile.to_dict(),
            "total_certificates_considered": len(certificates),
            "matches": [verdict.to_dict(cert) for cert, verdict in verdicts],
            "matched_certificates": len(verdicts),
            "matching_method": mode,
        }

    def _empty_response(self, tender: TenderDB, profile, total_certificates: int, mode: str) -> Dict[str, Any]:
        return {
            "match_id": None,
            "tender_id": tender.id,
            "tender_title": tender.title,
            "profile": profile.to_dict(),
            "total_certificates_considered": total_certificates,
            "matches": [],
            "matched_certificates": 0,
            "matching_method": mode,
        }

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        candidate = (mode or "keyword").strip().lower()
        if candidate not in {"keyword", "ai"}:
            return "keyword"
        return candidate

certificate_tender_matcher = CertificateTenderMatcher()
