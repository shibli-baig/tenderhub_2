"""
Tender Eligibility Processor Module for TenderHub.

This module processes tender documents to extract technical eligibility criteria
that require certificate proof. It uses GPT-4 Vision for page analysis and
OpenAI embeddings for semantic matching.
"""

import io
import os
import uuid
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import PyPDF2
from openai import OpenAI, RateLimitError

from database import (
    SessionLocal,
    TenderDB,
    TenderDocumentDB,
    TenderAnalysisStatusDB,
    TenderEligibilityCriteriaDB
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EXTRACTION_MODEL = os.getenv('OPENAI_EXTRACTION_MODEL', 'gpt-4o')  # Fixed: gpt-5 doesn't exist
EMBEDDINGS_MODEL = os.getenv('OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-large')

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✓ OpenAI client initialized for tender analysis")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.error("OPENAI_API_KEY not set")

TENDER_ELIGIBILITY_SYSTEM_PROMPT = (
    "You are an expert analyst for Indian government tenders. "
    "Your job is to identify only those eligibility or pre-qualification clauses that require the bidder "
    "to prove experience, capability, or certification with supporting documents such as completion certificates, "
    "similar work orders, technical personnel CVs, financial statements, or registrations. "
    "Always produce valid JSON and do not include explanatory prose."
)

TENDER_ELIGIBILITY_EXTRACTION_PROMPT = """You are reviewing the tender document named "{document_name}".

Extract ONLY the eligibility or pre-qualification criteria that require verifiable proof (e.g., completion certificates, similar work experience, technical personnel CVs, financial statements, registrations/licences).

Instructions:
- Ignore general scope descriptions, payment terms, bid submission steps, or clauses that do not demand proof of past performance/competence.
- Keep verbatim text exactly as in the document (preserve punctuation, case, numbering, and any Hindi or bilingual content).
- Categorise each criterion using one of: authority, scope_of_work, metrics, financial, technical, services, location, experience, other.
- Type should be "mandatory", "desirable" (nice-to-have), or "scoring" (evaluated for marks).
- `requirements` should capture structured data when possible, e.g. {{"min_value": 50, "unit": "km", "project_type": "road construction"}} or {{"similar_projects": 3, "lookback_years": 5}}.
- Collect 3-8 compact keywords per criterion for downstream semantic search.
- Include a `confidence` score between 0 and 1.
- If the document mentions several alternate thresholds (e.g., "Any 3 works of 40% OR 2 works of 50%"), split them into separate requirement objects inside the same criterion or represent them clearly in the requirements JSON.

Return ONLY a JSON object with this structure:
{{
  "has_eligibility_criteria": true | false,
  "summary": "One paragraph summary of the key eligibility expectations.",
  "criteria": [
    {{
      "category": "...",
      "type": "mandatory",
      "text": "Full verbatim clause...",
      "requirements": {{...}},
      "keywords": ["term1", "term2"],
      "confidence": 0.92
    }}
  ],
  "notes": "Any important caveats or observations about the eligibility section."
}}

Document Text:
---
{document_text}
---
"""

MAX_DOCUMENT_CHARS = int(os.getenv("TENDER_ANALYSIS_MAX_CHARS", "90000"))
MIN_DOCUMENT_CHARS = int(os.getenv("TENDER_ANALYSIS_MIN_CHARS", "12000"))
PROMPT_SHRINK_FACTOR = float(os.getenv("TENDER_ANALYSIS_SHRINK_FACTOR", "0.75"))


class TenderEligibilityProcessor:
    """Main class for processing tender documents and extracting eligibility criteria."""

    def __init__(self):
        """Initialize the tender eligibility processor."""
        self.embedding_model = EMBEDDINGS_MODEL

        # Dimension based on model
        if "3-large" in EMBEDDINGS_MODEL:
            self.dimension = 3072
        elif "3-small" in EMBEDDINGS_MODEL:
            self.dimension = 1536
        else:
            self.dimension = 1536

        logger.info(f"✓ TenderEligibilityProcessor initialized with {self.embedding_model}")

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract raw text from an entire PDF document."""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_chunks: List[str] = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text:
                    text_chunks.append(page_text)
            return "\n\n".join(text_chunks)
        except Exception as exc:
            logger.error(f"Unable to extract text from PDF: {exc}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text content using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not openai_client:
            raise ValueError("OpenAI client not initialized")

        try:
            # Filter valid texts
            valid_texts = [text.strip()[:8000] for text in texts if text and text.strip()]

            if not valid_texts:
                logger.debug("No valid texts to embed; skipping embeddings.")
                return []

            logger.debug(f"Generating embeddings for {len(valid_texts)} texts...")

            response = openai_client.embeddings.create(
                model=self.embedding_model,
                input=valid_texts
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Try to parse JSON, cleaning smart quotes or markdown fences when needed."""

        def _clean(text: str) -> str:
            return (
                text.replace("“", '"')
                    .replace("”", '"')
                    .replace("’", "'")
                    .replace("‘", "'")
            )

        stripped = raw_text.strip()
        if stripped.startswith("```json"):
            stripped = stripped[7:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            cleaned = _clean(stripped)
            return json.loads(cleaned)

    @staticmethod
    def _normalize_requirement_payload(payload: Any) -> Any:
        """Ensure requirements payload is JSON serialisable (dict/list)."""
        if payload is None:
            return {}
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return {}
            try:
                parsed = TenderEligibilityProcessor._parse_json_response(stripped)
                # Ensure the parsed result is dict or list, not a string or other type
                if isinstance(parsed, (dict, list)):
                    return parsed
                else:
                    # If parsed result is a string or other type, wrap it in a dict
                    return {"notes": str(parsed)}
            except Exception:
                return {"notes": stripped}
        # For any other type (int, float, bool, etc.), convert to dict
        return {"value": payload}

    def process_tender_eligibility(self, tender_id: str, analysis_status_id: Optional[str] = None) -> str:
        """
        Main pipeline to process a tender and extract eligibility criteria.

        Args:
            tender_id: UUID of the tender to process

        Returns:
            Analysis status ID
        """
        db = SessionLocal()
        analysis_status_id = analysis_status_id or str(uuid.uuid4())

        try:
            logger.info(f"Starting eligibility analysis for tender {tender_id}")

            analysis_status = db.query(TenderAnalysisStatusDB).filter(
                TenderAnalysisStatusDB.id == analysis_status_id
            ).first()

            if analysis_status is None:
                analysis_status = TenderAnalysisStatusDB(
                    id=analysis_status_id,
                    tender_id=tender_id,
                    status='processing',
                    started_at=datetime.utcnow()
                )
                db.add(analysis_status)
            else:
                analysis_status.status = 'processing'
                analysis_status.started_at = datetime.utcnow()
                analysis_status.completed_at = None
                analysis_status.error_message = None
                analysis_status.total_pages = 0
                analysis_status.processed_pages = 0
                analysis_status.total_criteria_found = 0
            db.commit()

            # Get tender and its documents
            tender = db.query(TenderDB).filter(TenderDB.id == tender_id).first()
            if not tender:
                raise ValueError(f"Tender {tender_id} not found")

            # Remove prior criteria if reprocessing
            db.query(TenderEligibilityCriteriaDB).filter(
                TenderEligibilityCriteriaDB.tender_id == tender_id
            ).delete(synchronize_session=False)
            db.commit()

            all_tender_docs = db.query(TenderDocumentDB).filter(
                TenderDocumentDB.tender_id == tender_id,
                TenderDocumentDB.document_type == 'pdf'
            ).all()

            if not all_tender_docs:
                raise ValueError(f"No PDF documents found for tender {tender_id}")

            tender_docs = [
                doc for doc in all_tender_docs
                if doc.file_data and 'boq' not in (doc.filename or '').lower()
            ]

            if not tender_docs:
                logger.warning("All available tender PDFs appear to be BOQ files; skipping analysis.")
                raise ValueError("No suitable tender documents found for eligibility extraction")

            largest_doc = max(tender_docs, key=lambda d: len(d.file_data or b''), default=None)
            if not largest_doc or not largest_doc.file_data:
                raise ValueError("Unable to locate tender document bytes for analysis")

            logger.info(f"Selected document '{largest_doc.filename}' ({len(largest_doc.file_data)} bytes) for eligibility extraction")

            document_text = self.extract_text_from_pdf(largest_doc.file_data)
            if not document_text.strip():
                raise ValueError("No textual content extracted from the tender PDF")

            prompt_text = document_text[:MAX_DOCUMENT_CHARS]
            if len(prompt_text) < MIN_DOCUMENT_CHARS:
                logger.info(
                    "Extracted text shorter than MIN_DOCUMENT_CHARS (%s); using entire payload (%s chars).",
                    MIN_DOCUMENT_CHARS,
                    len(prompt_text),
                )

            # Prepare status for single-pass processing
            analysis_status.total_pages = 1
            analysis_status.processed_pages = 0
            db.commit()
            if not openai_client:
                raise ValueError("OpenAI client not initialized")

            logger.info("Submitting tender document for eligibility extraction...")

            max_attempts = 4
            attempt = 0
            response = None
            last_error: Optional[Exception] = None

            while attempt < max_attempts:
                attempt += 1
                try:
                    formatted_prompt = TENDER_ELIGIBILITY_EXTRACTION_PROMPT.format(
                        document_name=largest_doc.filename,
                        document_text=prompt_text
                    )
                    logger.debug("Formatted prompt length (%s chars) on attempt %s/%s", len(formatted_prompt), attempt, max_attempts)
                except KeyError as key_err:
                    logger.error(f"Prompt formatting failed - missing key: {key_err}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise ValueError(f"Prompt template formatting error: {str(key_err)}")
                except Exception as fmt_exc:
                    logger.error(f"Unexpected prompt formatting error: {fmt_exc}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise ValueError(f"Prompt formatting failed: {str(fmt_exc)}")

                try:
                    response = openai_client.chat.completions.create(
                        model=EXTRACTION_MODEL,
                        temperature=0.1,
                        max_completion_tokens=10000,
                        messages=[
                            {"role": "system", "content": TENDER_ELIGIBILITY_SYSTEM_PROMPT},
                            {"role": "user", "content": formatted_prompt}
                        ]
                    )
                    logger.info("Eligibility extraction completed on attempt %s", attempt)
                    break
                except RateLimitError as api_exc:
                    last_error = api_exc
                    error_message = str(api_exc).lower()
                    needs_shrink = any(
                        phrase in error_message
                        for phrase in [
                            "tokens per min",
                            "request too large",
                            "rate limit",
                        ]
                    )
                    if not needs_shrink:
                        logger.error(f"OpenAI API call failed: {api_exc}")
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        raise ValueError(f"Failed to call OpenAI API: {str(api_exc)}")

                    if len(prompt_text) <= MIN_DOCUMENT_CHARS:
                        logger.error("Prompt already at minimum length (%s chars); cannot shrink further.", len(prompt_text))
                        raise ValueError(f"Failed to call OpenAI API: {str(api_exc)}")

                    new_length = int(len(prompt_text) * PROMPT_SHRINK_FACTOR)
                    new_length = max(MIN_DOCUMENT_CHARS, new_length)
                    if new_length >= len(prompt_text):
                        new_length = max(MIN_DOCUMENT_CHARS, len(prompt_text) - 1000)

                    if new_length <= 0 or new_length >= len(prompt_text):
                        logger.error("Unable to reduce prompt size further (current length %s).", len(prompt_text))
                        raise ValueError(f"Failed to call OpenAI API: {str(api_exc)}")

                    logger.warning(
                        "OpenAI reported oversized request (attempt %s/%s). Shrinking prompt from %s to %s characters.",
                        attempt,
                        max_attempts,
                        len(prompt_text),
                        new_length,
                    )
                    prompt_text = prompt_text[:new_length]
                    continue
                except Exception as api_exc:
                    last_error = api_exc
                    logger.error(f"OpenAI API call failed: {api_exc}")
                    logger.error(f"Error type: {type(api_exc).__name__}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise ValueError(f"Failed to call OpenAI API: {str(api_exc)}")

            if response is None:
                raise ValueError(f"Failed to call OpenAI API: {str(last_error) if last_error else 'Unknown error'}")

            logger.debug("Eligibility extraction response received from OpenAI.")

            try:
                result_raw = response.choices[0].message.content or ""
                logger.debug("Eligibility extraction raw content: %s", (result_raw or "")[:500])
            except (IndexError, AttributeError) as extract_exc:
                logger.error(f"Failed to extract content from API response: {extract_exc}")
                raise ValueError(f"Invalid API response structure: {str(extract_exc)}")

            try:
                result = self._parse_json_response(result_raw)
            except Exception as parse_exc:
                logger.error(f"Failed to parse eligibility response: {parse_exc}")
                logger.error(f"Raw response text: {result_raw[:500]}...")
                raise ValueError(f"Failed to parse GPT response as JSON: {str(parse_exc)}")

            criteria_payload = result.get("criteria", []) or []

            analysis_status.processed_pages = 1
            analysis_status.total_criteria_found = len(criteria_payload)
            db.commit()

            logger.info(f"Received {len(criteria_payload)} criteria from model.")
            sample_payload = criteria_payload[:1] if isinstance(criteria_payload, list) else criteria_payload
            logger.debug("Criteria payload sample: %s", sample_payload)
            if criteria_payload:
                criteria_texts = [c.get("text", "")[:4000] for c in criteria_payload if c.get("text")]
                embeddings = self.generate_embeddings(criteria_texts) if criteria_texts else []

                embedding_lookup = iter(embeddings)
                successfully_processed = 0
                for idx, criterion in enumerate(criteria_payload):
                    try:
                        criterion_text = (criterion.get("text") or "").strip()
                        if not criterion_text:
                            logger.warning(f"Skipping criterion {idx+1}: No text found")
                            continue

                        try:
                            embedding = next(embedding_lookup) if embeddings else None
                        except StopIteration:
                            embedding = None

                        requirements_payload = self._normalize_requirement_payload(criterion.get("requirements"))

                        # Ensure requirements_payload is always a dict or list
                        if not isinstance(requirements_payload, (dict, list)):
                            logger.warning(f"Criterion {idx+1}: Requirements normalized to unexpected type {type(requirements_payload)}, converting to dict")
                            requirements_payload = {"value": str(requirements_payload)}

                        keywords_payload = criterion.get("keywords", [])
                        if isinstance(keywords_payload, str):
                            keywords_payload = [keywords_payload]

                        criteria_record = TenderEligibilityCriteriaDB(
                            id=str(uuid.uuid4()),
                            tender_id=tender_id,
                            page_number=criterion.get("page_number") or 1,
                            category=criterion.get("category", "other"),
                            criteria_type=criterion.get("type", "mandatory"),
                            criteria_text=criterion_text,
                            extracted_requirements=requirements_payload,
                            keywords=keywords_payload,
                            embedding=embedding,
                            confidence_score=criterion.get("confidence", 0.0),
                            processing_status='completed'
                        )
                        db.add(criteria_record)
                        successfully_processed += 1

                    except Exception as criterion_exc:
                        logger.error(f"Failed to process criterion {idx+1}: {criterion_exc}")
                        logger.error(f"Problematic criterion data: {criterion}")
                        # Continue processing other criteria instead of failing entirely
                        continue

                logger.info(f"Successfully processed {successfully_processed}/{len(criteria_payload)} criteria")
                db.commit()

            analysis_status.status = 'completed'
            analysis_status.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"✓ Successfully completed analysis for tender {tender_id}")
            return analysis_status_id

        except Exception as e:
            logger.error(f"Failed to process tender {tender_id}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            # Update status to failed
            try:
                analysis_status = db.query(TenderAnalysisStatusDB).filter(
                    TenderAnalysisStatusDB.id == analysis_status_id
                ).first()

                if analysis_status:
                    analysis_status.status = 'failed'
                    analysis_status.error_message = str(e)
                    analysis_status.completed_at = datetime.utcnow()
                    db.commit()
            except Exception as db_exc:
                logger.error(f"Failed to update analysis status: {db_exc}")

            raise

        finally:
            db.close()


# Global processor instance
tender_eligibility_processor = TenderEligibilityProcessor()
