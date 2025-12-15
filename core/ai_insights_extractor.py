"""
AI Insights Extractor - Integrates CM5 extraction logic
Extracts PQ and Eligibility criteria from tender documents using GPT-4o
"""

import os
import logging
import tempfile
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# Import CM5 components (copied from CM5 directory)
from core.cm5_pdf_parser import CM5PDFParser
from core.cm5_extractor import CM5PQExtractor

# Import app components
from database import TenderDocumentDB, TenderAIInsightsDB

logger = logging.getLogger(__name__)


class AIInsightsExtractor:
    """Extracts AI insights from tender documents"""

    def __init__(self, openai_api_key: str):
        self.pdf_parser = CM5PDFParser()
        self.pq_extractor = CM5PQExtractor(openai_api_key)

    def extract_insights(
        self,
        tender_id: str,
        db: Session,
        force: bool = False,
        document_id: Optional[int] = None
    ) -> Dict:
        """
        Extract AI insights from tender document

        Args:
            tender_id: Tender ID
            db: Database session
            force: Force re-extraction even if exists
            document_id: Optional specific document ID to analyze (if None, uses largest PDF)

        Returns:
            Extraction result dictionary
        """

        # Check if insights already exist
        existing = db.query(TenderAIInsightsDB).filter(
            TenderAIInsightsDB.tender_id == tender_id
        ).first()

        if existing and not force:
            if existing.extraction_status == "completed":
                return {
                    "status": "completed",
                    "insights_id": existing.id,
                    "message": "Insights already extracted"
                }
            elif existing.extraction_status == "processing":
                return {
                    "status": "processing",
                    "insights_id": existing.id,
                    "message": "Extraction in progress"
                }

        # Get PDF document (either specified or largest)
        if document_id:
            # Use specified document
            document = db.query(TenderDocumentDB).filter(
                TenderDocumentDB.id == document_id,
                TenderDocumentDB.tender_id == tender_id,
                TenderDocumentDB.document_type == "pdf"
            ).first()
            if not document:
                raise ValueError(f"Document {document_id} not found or is not a PDF")
        else:
            # Use largest PDF document
            document = db.query(TenderDocumentDB).filter(
                TenderDocumentDB.tender_id == tender_id,
                TenderDocumentDB.document_type == "pdf"
            ).order_by(TenderDocumentDB.file_size.desc()).first()

            if not document:
                raise ValueError("No PDF documents found for this tender")

        # Create or update insights record
        if existing and force:
            insights_record = existing
            insights_record.extraction_status = "processing"
            insights_record.started_at = datetime.utcnow()
            insights_record.error_message = None
        else:
            insights_record = TenderAIInsightsDB(
                tender_id=tender_id,
                document_id=document.id,
                document_filename=document.filename,
                extraction_status="processing",
                started_at=datetime.utcnow()
            )
            db.add(insights_record)

        db.commit()

        try:
            # Save document to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(document.file_data)
                tmp_path = tmp_file.name

            # Extract text with page markers
            text_content = self.pdf_parser.extract_text(tmp_path)
            page_count = self.pdf_parser.get_page_count(tmp_path)

            # Validate extracted text
            if not text_content or text_content.strip() == "":
                raise ValueError(
                    f"PDF extraction resulted in empty text. This may be a scanned PDF without OCR. "
                    f"Document: {document.filename}, Pages: {page_count}"
                )

            if len(text_content) < 500:
                logger.warning(
                    f"Extracted text is very short ({len(text_content)} chars). "
                    f"Document may be empty or extraction incomplete. File: {document.filename}"
                )

            logger.info(f"Extracted {len(text_content)} chars from {page_count} pages of {document.filename}")

            # Extract criteria using GPT-4o
            extraction_result = self.pq_extractor.extract_criteria(text_content)

            # Check for errors in extraction result
            if 'error' in extraction_result and extraction_result['error']:
                logger.error(f"Extraction contained error: {extraction_result['error']}")
                # Store the error but still mark as completed (with error details)
                insights_record.error_message = extraction_result['error']

            # Update insights record
            insights_record.extraction_status = "completed"
            insights_record.completed_at = datetime.utcnow()
            insights_record.document_page_count = page_count
            insights_record.pq_criteria = extraction_result.get("pq_criteria", [])
            insights_record.eligibility_criteria = extraction_result.get("eligibility_criteria", [])
            insights_record.sections = extraction_result.get("sections", [])
            insights_record.summary = extraction_result.get("summary", {})

            # Store raw GPT response for transparency
            insights_record.raw_gpt_response = extraction_result.get("raw_response", "")

            # Update statistics
            summary = extraction_result.get("summary", {})
            insights_record.total_pq_criteria = summary.get("total_pq_criteria", 0)
            insights_record.total_eligibility_criteria = summary.get("total_eligibility_criteria", 0)
            insights_record.mandatory_count = summary.get("mandatory_count", 0)
            insights_record.optional_count = summary.get("optional_count", 0)

            # Store extraction quality metadata (Phase 3)
            original_text_length = len(text_content)
            insights_record.text_length = original_text_length
            insights_record.was_truncated = original_text_length > 80000

            # Build warnings list
            warnings = []
            if original_text_length < 500:
                warnings.append("Very short text extracted - may not contain sufficient criteria")
            if insights_record.was_truncated:
                warnings.append(f"Document text truncated from {original_text_length} to 80000 chars")
            if insights_record.total_pq_criteria + insights_record.total_eligibility_criteria == 0:
                warnings.append("No criteria extracted - document may not contain PQ/Eligibility sections")
            if extraction_result.get('error'):
                warnings.append(f"Extraction error: {extraction_result['error']}")

            insights_record.extraction_warnings = warnings

            if warnings:
                logger.warning(f"Extraction completed with {len(warnings)} warnings: {warnings}")

            db.commit()

            # Clean up temp file
            os.unlink(tmp_path)

            return {
                "status": "completed",
                "insights_id": insights_record.id,
                "total_criteria": len(extraction_result.get("pq_criteria", [])) + len(extraction_result.get("eligibility_criteria", [])),
                "extraction_result": extraction_result
            }

        except Exception as e:
            logger.error(f"Extraction failed for tender {tender_id}: {e}")
            insights_record.extraction_status = "failed"
            insights_record.completed_at = datetime.utcnow()
            insights_record.error_message = str(e)
            db.commit()

            # Clean up temp file if exists
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            raise
