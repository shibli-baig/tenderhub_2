"""
Certificate Processing Module for TenderHub.

This module handles PDF text extraction, OCR processing, OpenAI GPT parsing,
and vector embeddings for certificate documents.
"""

import os
import uuid
import json
import re
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
import anthropic
from anthropic import APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError, APITimeoutError as AnthropicAPITimeoutError, APIConnectionError as AnthropicAPIConnectionError
import numpy as np
import faiss
import base64
from io import BytesIO
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

from database import SessionLocal, CertificateDB, VectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose library loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Valid model names (as of Dec 2024)
VALID_OPENAI_MODELS = [
    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 
    'gpt-4', 'gpt-3.5-turbo', 'gpt-4o-2024-11-20',
    'gpt-4o-2024-08-06', 'gpt-4o-mini-2024-07-18'
]
VALID_CLAUDE_MODELS = [
    'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022',
    'claude-3-opus-20240229', 'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307'
]

def validate_model_name(model_name: str, model_type: str = 'openai') -> str:
    """
    Validate and return a safe model name.
    
    Args:
        model_name: Model name to validate
        model_type: 'openai' or 'claude'
    
    Returns:
        Valid model name (returns safe default if invalid)
    """
    valid_models = VALID_OPENAI_MODELS if model_type == 'openai' else VALID_CLAUDE_MODELS
    safe_default = 'gpt-4o' if model_type == 'openai' else 'claude-sonnet-4-20250514'
    
    if model_name not in valid_models:
        logger.warning(
            f"⚠️ Invalid {model_type.upper()} model '{model_name}'. "
            f"Valid models: {', '.join(valid_models[:3])}... "
            f"Using safe default: {safe_default}"
        )
        return safe_default
    
    return model_name

# Load and validate extraction model
EXTRACTION_MODEL_RAW = os.getenv('OPENAI_EXTRACTION_MODEL', 'gpt-4o')
EXTRACTION_MODEL = validate_model_name(EXTRACTION_MODEL_RAW, 'openai')

EXTRACTION_MAX_TOKENS = int(os.getenv('OPENAI_EXTRACTION_MAX_TOKENS', '4000'))
EXTRACTION_TEMPERATURE = float(os.getenv('OPENAI_EXTRACTION_TEMPERATURE', '0.1'))
EMBEDDINGS_MODEL = os.getenv('OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-large')  # Best quality, supports Hindi

# Initialize OpenAI clients only if API key is available
openai_client = None
openai_wrapper = None

if OPENAI_API_KEY:
    try:
        from core.openai_wrapper import OpenAIWrapper
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        openai_wrapper = OpenAIWrapper(api_key=OPENAI_API_KEY)
        logger.info("✓ OpenAI client initialized successfully")
    except Exception as e:
        logger.warning(f"⚠ OpenAI client initialization failed: {e}")
else:
    logger.warning("⚠ OPENAI_API_KEY not set. Certificate processing features will be limited.")


# Anthropic (Claude) configuration for fallback extraction
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_EXTRACTION_MODEL_RAW = os.getenv('CLAUDE_EXTRACTION_MODEL', 'claude-sonnet-4-20250514')
CLAUDE_EXTRACTION_MODEL = validate_model_name(CLAUDE_EXTRACTION_MODEL_RAW, 'claude')

# Toggle for Claude alternation: True = alternate GPT/Claude, False = GPT only
USE_CLAUDE_ALTERNATION = os.getenv('USE_CLAUDE_ALTERNATION', 'true').lower() in ('true', '1', 'yes', 'on')

# Batch processing configuration
BATCH_DELAY_SECONDS = float(os.getenv('BATCH_DELAY_SECONDS', '0.5'))
MAX_CONCURRENT_WORKERS = int(os.getenv('MAX_CONCURRENT_WORKERS', '3'))

# Initialize Anthropic client
anthropic_client = None

if ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info(f"✓ Anthropic client initialized successfully (Model: {CLAUDE_EXTRACTION_MODEL})")
        logger.info(f"  Claude alternation is {'ENABLED' if USE_CLAUDE_ALTERNATION else 'DISABLED'}")
    except Exception as e:
        logger.warning(f"⚠ Anthropic client initialization failed: {e}")
else:
    if USE_CLAUDE_ALTERNATION:
        logger.warning("⚠ ANTHROPIC_API_KEY not set. Claude alternation will not be available.")

# Log active configuration
logger.info(f"📋 Certificate Processor Configuration:")
logger.info(f"  • OpenAI Model: {EXTRACTION_MODEL}")
logger.info(f"  • Claude Model: {CLAUDE_EXTRACTION_MODEL}")
logger.info(f"  • Alternation: {'ON' if USE_CLAUDE_ALTERNATION and anthropic_client else 'OFF'}")
logger.info(f"  • Batch Delay: {BATCH_DELAY_SECONDS}s")
logger.info(f"  • Max Workers: {MAX_CONCURRENT_WORKERS}")


# Retry configuration for OpenAI API calls
OPENAI_MAX_RETRIES = int(os.getenv('OPENAI_MAX_RETRIES', '5'))
OPENAI_RETRY_MIN_WAIT = int(os.getenv('OPENAI_RETRY_MIN_WAIT', '2'))
OPENAI_RETRY_MAX_WAIT = int(os.getenv('OPENAI_RETRY_MAX_WAIT', '60'))


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry."""
    if isinstance(exception, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exception, APIError):
        # Retry on 5xx server errors
        if hasattr(exception, 'status_code') and exception.status_code >= 500:
            return True
    return False


@retry(
    stop=stop_after_attempt(OPENAI_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=OPENAI_RETRY_MIN_WAIT, max=OPENAI_RETRY_MAX_WAIT),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_openai_with_retry(**request_kwargs):
    """
    Make OpenAI API call with automatic retry on rate limits and transient errors.
    """
    return openai_client.chat.completions.create(**request_kwargs)


def _call_chat_completion_with_temperature_fallback(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    **kwargs: Any
):
    """
    Call OpenAI chat completions API with graceful fallback when temperature is unsupported.
    Includes automatic retry with exponential backoff for rate limits and transient errors.

    Some newer models (e.g. reasoning models) only allow the default temperature=1.0.
    When we encounter that error, retry once without the temperature parameter.
    """
    if not openai_client:
        raise ValueError("OpenAI API key not configured")

    request_kwargs = {
        "model": model,
        "messages": messages,
        **kwargs,
    }

    if temperature is not None:
        request_kwargs["temperature"] = temperature

    try:
        # Use retry-wrapped function
        return _call_openai_with_retry(**request_kwargs)
    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        # If we've exhausted retries, log and re-raise
        logger.error(f"OpenAI API call failed after {OPENAI_MAX_RETRIES} retries: {e}")
        raise
    except Exception as exc:
        error_text = str(exc)
        temp_was_set = temperature is not None
        if temp_was_set and "temperature" in error_text and "Unsupported value" in error_text:
            logger.warning(
                "Model %s rejected temperature %.3f; retrying with default temperature.",
                model,
                temperature,
            )
            safe_kwargs = dict(request_kwargs)
            safe_kwargs.pop("temperature", None)
            return _call_openai_with_retry(**safe_kwargs)
        raise


# Retry configuration for Anthropic API calls
@retry(
    stop=stop_after_attempt(OPENAI_MAX_RETRIES),  # Reuse same retry count
    wait=wait_exponential(multiplier=1, min=OPENAI_RETRY_MIN_WAIT, max=OPENAI_RETRY_MAX_WAIT),
    retry=retry_if_exception_type((AnthropicRateLimitError, AnthropicAPITimeoutError, AnthropicAPIConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_anthropic_with_retry(**request_kwargs):
    """
    Make Anthropic API call with automatic retry on rate limits and transient errors.
    """
    return anthropic_client.messages.create(**request_kwargs)


# FAISS index configuration
FAISS_INDEX_DIR = "faiss_indices"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Simplified schema we expect from GPT extraction
CERTIFICATE_JSON_TEMPLATE = {
    "project_name": None,
    "client_name": None,
    "location": None,
    "location_original_script": None,
    "services_rendered": [],
    "sectors": [],
    "sub_sectors": [],
    "scope_of_work": None,
    "project_value_inr": None,
    "consultancy_fee_inr": None,
    "start_date": None,
    "end_date": None,
    "completion_date": None,
    "duration": None,
    "issuing_authority_details": None,
    "signing_authority_details": None,
    "certificate_number": None,
    "performance_remarks": None,
    "role_lead_jv": None,
    "jv_partners": [],
    "funding_agency": None,
    "metrics": [],
    "verbatim_certificate": None,
    "confidence_score": 0.0
}

CERTIFICATE_EXTRACTION_PROMPT = f"""
You are a meticulous analyst who reads government/commercial completion certificates from India.

RULES:
1. Read ALL provided pages/images - they all belong to the same certificate.
2. Extract information exactly as printed. Keep original Hindi/English text; do not translate unless specifically asked.
3. Convert Hindi numerals (०१२३४५६७८९) to standard numerals inside numeric fields.
4. Always respond with a STRICT JSON object that matches this template exactly:
{json.dumps(CERTIFICATE_JSON_TEMPLATE, indent=2)}
5. Use null when information is missing. Arrays must always be arrays (even if empty).
6. `verbatim_certificate` must contain the cleaned full text from the document so we can reparse later.
7. `confidence_score` must be between 0 and 1 and reflect your confidence in the extracted data.
8. Do not include any commentary outside the JSON.
"""

CERTIFICATE_MAX_PAGES = int(os.getenv("CERTIFICATE_MAX_PAGES", "6"))
CERTIFICATE_EXTRACTION_ATTEMPTS = int(os.getenv("CERTIFICATE_EXTRACTION_ATTEMPTS", "2"))


def parse_consultancy_fee(fee_str: str) -> Optional[float]:
    """
    Parse consultancy fee string to numeric value with robust error handling.

    Handles various formats:
    - ₹5,00,000
    - Rs. 2.5 Crore
    - 150 Lakh
    - Plain numbers

    Args:
        fee_str: Fee string with currency symbols and Indian number formatting

    Returns:
        Float value in INR, or None if parsing fails
    """
    if not fee_str or not isinstance(fee_str, str):
        return None

    fee_str = fee_str.strip()
    
    def safe_float_convert(value_str: str, multiplier: float = 1.0) -> Optional[float]:
        """Safely convert string to float with validation."""
        try:
            # Remove commas
            cleaned = value_str.replace(',', '')
            
            # Validate: should have at most one decimal point
            if cleaned.count('.') > 1:
                logger.warning(f"⚠️ Invalid numeric format (multiple decimals): '{value_str}' - skipping")
                return None
            
            # Validate: should not start with decimal point
            if cleaned.startswith('.'):
                logger.warning(f"⚠️ Invalid numeric format (leading decimal): '{value_str}' - skipping")
                return None
            
            # Validate: should contain only digits and at most one decimal point
            if not re.match(r'^\d+\.?\d*$', cleaned):
                logger.warning(f"⚠️ Invalid numeric format: '{value_str}' - skipping")
                return None
            
            return float(cleaned) * multiplier
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"⚠️ Float conversion failed for '{value_str}': {e}")
            return None

    # Handle Crore notation (1 Crore = 10,000,000)
    crore_match = re.search(r'([\d,\.]+)\s*(?:cr|crore)', fee_str, re.IGNORECASE)
    if crore_match:
        return safe_float_convert(crore_match.group(1), 10000000)

    # Handle Lakh notation (1 Lakh = 100,000)
    lakh_match = re.search(r'([\d,\.]+)\s*(?:l|lakh)', fee_str, re.IGNORECASE)
    if lakh_match:
        return safe_float_convert(lakh_match.group(1), 100000)

    # Handle plain numbers with commas
    num_match = re.search(r'[\d,]+(?:\.\d+)?', fee_str)
    if num_match:
        return safe_float_convert(num_match.group(0))

    return None


def validate_extraction_quality(parsed_data: Dict[str, Any], extracted_text: str) -> Tuple[bool, str]:
    """
    Validate if extraction quality is acceptable.
    
    Args:
        parsed_data: Parsed certificate data dictionary
        extracted_text: Raw extracted text from certificate
        
    Returns:
        Tuple of (is_valid, reason) where is_valid is bool and reason explains why
    """
    # Check if extracted text is meaningful
    if len(extracted_text) < 100:
        return False, "Extracted text too short (< 100 chars)"
    
    # Check for garbled text (high ratio of special characters)
    # This catches cases where OCR produces mostly gibberish
    if len(extracted_text) > 0:
        special_char_ratio = sum(1 for c in extracted_text if not c.isalnum() and not c.isspace()) / len(extracted_text)
        if special_char_ratio > 0.5:
            return False, "Text appears corrupted/garbled (>50% special chars)"
    
    # Count "NOT FOUND" or empty critical fields
    critical_fields = ['project_name', 'client_name', 'location']
    missing_count = 0
    for field in critical_fields:
        value = parsed_data.get(field)
        if not value or value in ['NOT FOUND', 'Not Found', 'NOT_FOUND', 'Not found', 'N/A', 'NA']:
            missing_count += 1
    
    if missing_count >= 2:  # At least 2 of 3 critical fields missing
        return False, f"Too many critical fields missing ({missing_count}/3: project_name, client_name, location)"
    
    # Check confidence score if available
    confidence = parsed_data.get('confidence_score')
    if confidence is not None and isinstance(confidence, (int, float)):
        if confidence < 0.3:
            return False, f"Confidence score too low ({confidence:.2f} < 0.3)"
    
    return True, "Quality acceptable"


class CertificateProcessor:
    """Main class for processing certificate documents with GPT/Claude alternation."""
    
    # Class-level counter for alternating between GPT and Claude
    _extraction_counter = 0
    _counter_lock = threading.Lock()

    def __init__(self):
        # Use OpenAI embeddings exclusively for better Hindi support
        self.embedding_model = EMBEDDINGS_MODEL

        # text-embedding-3-large has 3072 dimensions
        if "3-large" in EMBEDDINGS_MODEL:
            self.dimension = 3072
        elif "3-small" in EMBEDDINGS_MODEL:
            self.dimension = 1536
        else:  # ada-002 fallback
            self.dimension = 1536

        logger.info(f"✓ Using OpenAI embeddings: {self.embedding_model} (supports 100+ languages including Hindi)")
    
    @classmethod
    def _get_next_model_preference(cls) -> str:
        """
        Get next model preference using round-robin alternation for load balancing.
        Returns: 'gpt' or 'claude'
        """
        with cls._counter_lock:
            cls._extraction_counter += 1
            # Alternate: odd=GPT, even=Claude (if available)
            if cls._extraction_counter % 2 == 1 or not (USE_CLAUDE_ALTERNATION and anthropic_client):
                return 'gpt'
            return 'claude'

    def _load_document_images(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Convert the uploaded document into a list of image payloads that can be sent to GPT-4o.
        Supports PDFs (converted page-by-page) and direct image uploads.
        """
        extension = Path(file_path).suffix.lower()
        payloads: List[Dict[str, Any]] = []

        if extension == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as exc:
                raise ImportError("pdf2image is required for PDF processing. Install with `pip install pdf2image`.") from exc

            pages = convert_from_path(file_path, dpi=300)
            if not pages:
                raise ValueError("PDF did not contain any renderable pages")

            for index, page in enumerate(pages[:CERTIFICATE_MAX_PAGES]):
                buffered = BytesIO()
                page.save(buffered, format="PNG")
                encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
                payloads.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                        "detail": "high"
                    }
                })
                logger.debug(f"Prepared PDF page {index+1} for GPT extraction")
        elif extension in {".jpg", ".jpeg", ".png"}:
            with open(file_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            image_type = "jpeg" if extension in {".jpg", ".jpeg"} else "png"
            payloads.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{encoded}",
                    "detail": "high"
                }
            })
            logger.debug("Prepared image file for GPT extraction")
        else:
            raise ValueError(f"Unsupported file type for certificate processing: {extension}")

        return payloads

    def _request_certificate_json(self, image_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call GPT-4o with the prepared image payload and return parsed JSON."""
        if not OPENAI_API_KEY or not openai_client:
            raise ValueError("OpenAI API key not configured")

        messages = [
            {"role": "system", "content": "You convert tender certificates into clean JSON. Output only valid JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CERTIFICATE_EXTRACTION_PROMPT},
                    *image_payload
                ]
            }
        ]

        response = _call_chat_completion_with_temperature_fallback(
            model=EXTRACTION_MODEL,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=6000
        )
        raw_response = response.choices[0].message.content or ""
        return self._extract_json(raw_response)

    def _request_certificate_json_claude(self, image_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call Claude with the prepared image payload and return parsed JSON.
        Uses the same prompt as GPT for consistency.
        """
        if not ANTHROPIC_API_KEY or not anthropic_client:
            raise ValueError("Anthropic API key not configured")

        # Convert GPT-style image payload to Claude format
        claude_content = [{"type": "text", "text": CERTIFICATE_EXTRACTION_PROMPT}]

        for img in image_payload:
            if img.get("type") == "image_url":
                url = img.get("image_url", {}).get("url", "")
                # Extract base64 data and media type from data URL
                if url.startswith("data:"):
                    # Format: data:image/png;base64,<data>
                    parts = url.split(",", 1)
                    if len(parts) == 2:
                        media_info = parts[0].replace("data:", "").replace(";base64", "")
                        base64_data = parts[1]
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_info,
                                "data": base64_data
                            }
                        })

        try:
            response = _call_anthropic_with_retry(
                model=CLAUDE_EXTRACTION_MODEL,
                max_tokens=6000,
                messages=[
                    {
                        "role": "user",
                        "content": claude_content
                    }
                ],
                system="You convert tender certificates into clean JSON. Output only valid JSON."
            )

            raw_response = response.content[0].text if response.content else ""
            return self._extract_json(raw_response)

        except (AnthropicRateLimitError, AnthropicAPITimeoutError, AnthropicAPIConnectionError) as e:
            logger.error(f"Claude API call failed after retries: {e}")
            raise
        except Exception as e:
            logger.error(f"Claude extraction error: {e}")
            raise

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Parse raw model output into JSON."""
        if not text:
            raise ValueError("Model returned empty response")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _normalize_string(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = str(value)
        value = str(value).strip()
        return value or None

    def _normalize_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            cleaned = []
            for item in value:
                normalized = self._normalize_string(item)
                if normalized:
                    cleaned.append(normalized)
            return cleaned
        if isinstance(value, str):
            parts = re.split(r"[,\n;]+", value)
            return [part.strip() for part in parts if part.strip()]
        return []

    def _normalize_metrics(self, value: Any) -> List[Dict[str, Optional[str]]]:
        metrics: List[Dict[str, Optional[str]]] = []
        if not value:
            return metrics
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    metrics.append({
                        "metric_name": self._normalize_string(item.get("metric_name")),
                        "value": self._normalize_string(item.get("value")),
                        "unit": self._normalize_string(item.get("unit")),
                        "notes": self._normalize_string(item.get("notes"))
                    })
                else:
                    normalized = self._normalize_string(item)
                    if normalized:
                        metrics.append({
                            "metric_name": normalized,
                            "value": None,
                            "unit": None,
                            "notes": None
                        })
        return metrics

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all expected keys exist with normalized values."""
        normalized: Dict[str, Any] = {}
        for key in CERTIFICATE_JSON_TEMPLATE.keys():
            normalized[key] = payload.get(key)

        normalized["project_name"] = self._normalize_string(normalized.get("project_name")) or "Certificate"
        normalized["client_name"] = self._normalize_string(normalized.get("client_name"))
        normalized["location"] = self._normalize_string(normalized.get("location"))
        normalized["location_original_script"] = self._normalize_string(normalized.get("location_original_script"))
        normalized["scope_of_work"] = self._normalize_string(normalized.get("scope_of_work"))
        normalized["project_value_inr"] = self._normalize_string(normalized.get("project_value_inr"))
        normalized["consultancy_fee_inr"] = self._normalize_string(normalized.get("consultancy_fee_inr"))
        normalized["start_date"] = self._normalize_string(normalized.get("start_date"))
        normalized["end_date"] = self._normalize_string(normalized.get("end_date"))
        normalized["completion_date"] = self._normalize_string(normalized.get("completion_date"))
        normalized["duration"] = self._normalize_string(normalized.get("duration"))
        normalized["issuing_authority_details"] = self._normalize_string(normalized.get("issuing_authority_details"))
        normalized["signing_authority_details"] = self._normalize_string(normalized.get("signing_authority_details"))
        normalized["certificate_number"] = self._normalize_string(normalized.get("certificate_number"))
        normalized["performance_remarks"] = self._normalize_string(normalized.get("performance_remarks"))
        normalized["role_lead_jv"] = self._normalize_string(normalized.get("role_lead_jv"))
        normalized["funding_agency"] = self._normalize_string(normalized.get("funding_agency"))
        normalized["verbatim_certificate"] = self._normalize_string(normalized.get("verbatim_certificate"))
        normalized["services_rendered"] = self._normalize_list(normalized.get("services_rendered"))
        normalized["sectors"] = self._normalize_list(normalized.get("sectors"))
        normalized["sub_sectors"] = self._normalize_list(normalized.get("sub_sectors"))
        normalized["jv_partners"] = self._normalize_list(normalized.get("jv_partners"))
        normalized["metrics"] = self._normalize_metrics(normalized.get("metrics"))

        try:
            confidence = float(payload.get("confidence_score", 0.8))
            normalized["confidence_score"] = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            normalized["confidence_score"] = 0.8

        return normalized

    @retry(
        stop=stop_after_attempt(OPENAI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=OPENAI_RETRY_MIN_WAIT, max=OPENAI_RETRY_MAX_WAIT),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _call_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Internal method to call embeddings API with retry."""
        response = openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [data.embedding for data in response.data]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text content using OpenAI API.
        Supports 100+ languages including Hindi with excellent semantic understanding.
        Includes automatic retry with exponential backoff for rate limits.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # Filter and validate input texts
        valid_texts = []
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                cleaned_text = text.strip()[:8000]  # Limit text length for API
                valid_texts.append(cleaned_text)

        if not valid_texts:
            raise ValueError("No valid text content to embed")

        if not OPENAI_API_KEY or not openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            logger.info(f"Generating {len(valid_texts)} embeddings via OpenAI {self.embedding_model}")

            # Use retry-wrapped method
            embeddings = self._call_embeddings_with_retry(valid_texts)

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            logger.error(f"Embedding generation failed after {OPENAI_MAX_RETRIES} retries: {e}")
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            logger.error(f"Input texts lengths: {[len(t) for t in valid_texts]}")
            raise

    def generate_embeddings_safe(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings with enhanced input validation and error handling.
        Prevents API errors from invalid characters or malformed inputs.
        
        Args:
            texts: List of text strings to embed (may contain problematic characters)
            
        Returns:
            List of embedding vectors
        """
        # Sanitize inputs thoroughly
        clean_texts = []
        for idx, text in enumerate(texts):
            if not text or not isinstance(text, str):
                clean_texts.append("No data available")
                logger.debug(f"Text {idx}: Empty or non-string, using placeholder")
                continue
            
            # Remove invalid characters that cause OpenAI API errors
            cleaned = text.strip()
            
            # Remove null bytes and replacement characters
            cleaned = cleaned.replace('\x00', '')  # Null bytes
            cleaned = cleaned.replace('\ufffd', '')  # Unicode replacement character
            cleaned = cleaned.replace('\u0000', '')  # Another null variant
            
            # Remove other control characters except newlines and tabs
            cleaned = ''.join(char for char in cleaned 
                            if char == '\n' or char == '\t' or ord(char) >= 32 or ord(char) == 9)
            
            # Normalize whitespace
            cleaned = ' '.join(cleaned.split())
            
            # Limit length to API maximum
            cleaned = cleaned[:8000]
            
            if len(cleaned) < 10:
                clean_texts.append("No data available")
                logger.debug(f"Text {idx}: Too short after cleaning ({len(cleaned)} chars), using placeholder")
            else:
                clean_texts.append(cleaned)
                logger.debug(f"Text {idx}: Cleaned and ready ({len(cleaned)} chars)")
        
        # Use the standard embedding generation method
        try:
            return self.generate_embeddings(clean_texts)
        except Exception as e:
            logger.error(f"Safe embedding generation failed even after sanitization: {e}")
            # Last resort: use generic placeholders for all texts
            logger.warning("Falling back to generic placeholder embeddings")
            fallback_texts = ["Certificate data" for _ in texts]
            return self.generate_embeddings(fallback_texts)

    def create_faiss_index(self, embeddings: List[List[float]]) -> str:
        """
        Create a FAISS index for the given embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Path to the saved FAISS index
        """
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Create FAISS index
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
            index.add(embeddings_array) # type: ignore

            # Save index
            index_id = str(uuid.uuid4())
            index_path = os.path.join(FAISS_INDEX_DIR, f"{index_id}.faiss")
            faiss.write_index(index, index_path)

            return index_path

        except Exception as e:
            logger.error(f"FAISS index creation failed: {e}")
            raise


    def process_certificate(
        self,
        user_id: str,
        file_path: str,
        filename: str,
        file_hash: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> str:
        """
        Complete certificate processing pipeline.

        Args:
            user_id: ID of the user uploading the certificate
            file_path: Path to the uploaded file
            filename: Original filename
            file_hash: Optional SHA256 hash for duplicate detection
            file_size: Optional file size in bytes

        Returns:
            Certificate ID
        """
        db = SessionLocal()
        # Initialize variables early to avoid undefined variable errors
        attempts_used = 0
        extraction_model_used = EXTRACTION_MODEL
        certificate_id = str(uuid.uuid4())

        try:
            logger.info(f"📄 Preparing document for extraction: {filename}")
            
            # Verify file exists before processing
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Certificate file not found at path: {file_path}")
            
            # Verify file is readable
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read certificate file: {file_path}")
            
            image_payload = self._load_document_images(file_path)

            # Determine if Claude alternation is available
            claude_available = USE_CLAUDE_ALTERNATION and ANTHROPIC_API_KEY and anthropic_client
            
            # Get model preference for this certificate (alternates across batch)
            preferred_model = self._get_next_model_preference()

            if claude_available:
                logger.info(f"📊 Sending {len(image_payload)} page(s) for extraction (alternating strategy: start with {preferred_model.upper()})")
            else:
                logger.info(f"📊 Sending {len(image_payload)} page(s) to GPT for structured extraction")

            normalized_payload: Optional[Dict[str, Any]] = None
            extracted_text = ""
            validation_reason = ""
            # attempts_used and extraction_model_used already initialized above

            # Build extraction sequence with preferred model first for load balancing
            # If preferred is GPT: GPT1 → Claude1 → GPT2 → Claude2
            # If preferred is Claude: Claude1 → GPT1 → Claude2 → GPT2
            extraction_sequence = []
            for i in range(1, CERTIFICATE_EXTRACTION_ATTEMPTS + 1):
                if preferred_model == 'gpt':
                    extraction_sequence.append(('gpt', i))
                    if claude_available:
                        extraction_sequence.append(('claude', i))
                else:  # preferred_model == 'claude'
                    if claude_available:
                        extraction_sequence.append(('claude', i))
                    extraction_sequence.append(('gpt', i))

            for model_type, attempt_num in extraction_sequence:
                try:
                    start_time = time.time()
                    
                    if model_type == 'gpt':
                        logger.info(f"🔄 Attempt {attempt_num} with GPT ({EXTRACTION_MODEL})...")
                        raw_payload = self._request_certificate_json(image_payload)
                        extraction_model_used = EXTRACTION_MODEL
                    else:  # claude
                        logger.info(f"🔄 Attempt {attempt_num} with Claude ({CLAUDE_EXTRACTION_MODEL})...")
                        raw_payload = self._request_certificate_json_claude(image_payload)
                        extraction_model_used = CLAUDE_EXTRACTION_MODEL
                    
                    response_time = time.time() - start_time
                    logger.info(f"⏱️ {model_type.upper()} response time: {response_time:.2f}s")

                    normalized_payload = self._normalize_payload(raw_payload)
                    extracted_text = normalized_payload.get("verbatim_certificate") or json.dumps(raw_payload, ensure_ascii=False)

                    is_valid, validation_reason = validate_extraction_quality(normalized_payload, extracted_text)
                    if is_valid:
                        logger.info(f"✅ Extraction validated using {model_type.upper()} on attempt {attempt_num} | Quality: {normalized_payload.get('confidence_score', 0):.2f}")
                        attempts_used = len([x for x in extraction_sequence[:extraction_sequence.index((model_type, attempt_num)) + 1]])
                        break

                    logger.warning(f"❌ {model_type.upper()} attempt {attempt_num} failed validation: {validation_reason}")
                    normalized_payload = None

                except Exception as e:
                    response_time = time.time() - start_time if 'start_time' in locals() else 0
                    logger.error(f"❌ {model_type.upper()} attempt {attempt_num} failed after {response_time:.2f}s | Error: {str(e)[:200]}")
                    normalized_payload = None
                    continue

            if not normalized_payload:
                raise ValueError(f"Unable to extract reliable data from certificate after all attempts: {validation_reason}")

            project_name = normalized_payload["project_name"] or f"Certificate {filename}"
            services = normalized_payload.get("services_rendered") or []
            sectors = normalized_payload.get("sectors") or []
            sub_sectors = normalized_payload.get("sub_sectors") or []
            jv_partners = normalized_payload.get("jv_partners") or []
            metrics = normalized_payload.get("metrics") or []

            completion_date = self._parse_date(normalized_payload.get("completion_date")) or self._parse_date(normalized_payload.get("end_date"))
            start_date_obj = self._parse_date(normalized_payload.get("start_date"))
            end_date_obj = self._parse_date(normalized_payload.get("end_date"))

            fee_string = normalized_payload.get("consultancy_fee_inr")
            fee_numeric = parse_consultancy_fee(fee_string) if fee_string else None
            project_value_raw = normalized_payload.get("project_value_inr")
            project_value_numeric = parse_consultancy_fee(project_value_raw) if project_value_raw else None

            certificate = CertificateDB(
                id=certificate_id,
                user_id=user_id,
                project_name=project_name,
                client_name=normalized_payload.get("client_name"),
                completion_date=completion_date or end_date_obj,
                project_value=project_value_numeric,
                project_value_inr=project_value_raw,
                services_rendered=services,
                location=normalized_payload.get("location"),
                sectors=sectors,
                sub_sectors=sub_sectors,
                consultancy_fee_inr=fee_string,
                consultancy_fee_numeric=fee_numeric,
                scope_of_work=normalized_payload.get("scope_of_work"),
                start_date=start_date_obj,
                end_date=end_date_obj,
                duration=normalized_payload.get("duration"),
                issuing_authority_details=normalized_payload.get("issuing_authority_details"),
                performance_remarks=normalized_payload.get("performance_remarks"),
                certificate_number=normalized_payload.get("certificate_number"),
                signing_authority_details=normalized_payload.get("signing_authority_details"),
                role_lead_jv=normalized_payload.get("role_lead_jv"),
                jv_partners=jv_partners,
                funding_agency=normalized_payload.get("funding_agency"),
                confidence_score=normalized_payload.get("confidence_score"),
                metrics=metrics,
                verbatim_certificate=normalized_payload.get("verbatim_certificate"),
                original_filename=filename,
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
                extracted_text=extracted_text,
                processing_status="completed",
                extraction_method="vision_direct",
                parsing_method=f"{extraction_model_used}_json",
                extraction_quality_score=normalized_payload.get("confidence_score"),
                extraction_attempts=attempts_used or 1,
                processed_at=datetime.utcnow()
            )

            db.add(certificate)
            db.commit()

            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for certificate {certificate_id}")

            # Create different text representations for embedding
            # Filter out empty strings and ensure valid text content
            client_name_value = normalized_payload.get('client_name', '')
            location_value = normalized_payload.get('location', '')
            scope_text = normalized_payload.get('scope_of_work') or "Scope not specified"
            authority_text = normalized_payload.get('issuing_authority_details') or normalized_payload.get('signing_authority_details') or "Authority not specified"
            metrics_text_parts = []
            for metric in metrics:
                name = metric.get('metric_name') or ''
                value = metric.get('value') or ''
                unit = metric.get('unit') or ''
                descriptor = " ".join(part for part in [name, value, unit] if part).strip()
                if descriptor:
                    metrics_text_parts.append(descriptor)
            metrics_text = "; ".join(metrics_text_parts) if metrics_text_parts else "Metrics not specified"

            services_text = " | ".join(services) if services else "Services not specified"
            sectors_text = " | ".join(filter(None, sectors + sub_sectors)) if (sectors or sub_sectors) else "Sectors not specified"
            jv_text = " | ".join(jv_partners) if jv_partners else "JV partners not specified"
            funding_text = normalized_payload.get('funding_agency') or "Funding agency not specified"

            full_text_embedding = extracted_text.strip() if extracted_text and extracted_text.strip() else f"Certificate content for {filename}"
            project_name_text = project_name.strip() if project_name and project_name.strip() else "Project name not specified"
            client_name_text = client_name_value.strip() if client_name_value and client_name_value.strip() else "Client name not specified"
            location_text = location_value.strip() if location_value and location_value.strip() else "Location not specified"

            embeddings_payload = [
                (full_text_embedding, 'full_text'),
                (project_name_text, 'project_name'),
                (client_name_text, 'client_name'),
                (services_text, 'services'),
                (location_text, 'location'),
                (sectors_text, 'sectors'),
                (scope_text, 'scope_of_work'),
                (authority_text, 'authority_details'),
                (metrics_text, 'metrics'),
                (jv_text, 'jv_partners'),
                (funding_text, 'funding_agency'),
            ]

            texts_to_embed = [text if text.strip() else f"{content_type.replace('_', ' ')} not specified"
                              for text, content_type in embeddings_payload]

            # Use safe embedding generation to prevent API errors
            embeddings = self.generate_embeddings_safe(texts_to_embed)

            # Step 5: Store embeddings and create FAISS index
            for embedding, (_, content_type) in zip(embeddings, embeddings_payload):
                vector = VectorDB(
                    certificate_id=certificate_id,
                    embedding=embedding,
                    content_type=content_type,
                    embedding_model=self.embedding_model
                )
                db.add(vector)

            # Create FAISS index for full text embedding
            faiss_index_path = self.create_faiss_index([embeddings[0]])  # Full text embedding

            # Update the first vector with FAISS index path
            db.query(VectorDB).filter(
                VectorDB.certificate_id == certificate_id,
                VectorDB.content_type == 'full_text'
            ).update({"faiss_index_path": faiss_index_path})

            db.commit()

            logger.info(f"Certificate {certificate_id} processed successfully")
            return certificate_id

        except Exception as e:
            logger.error(f"Certificate processing failed: {e}")
            db.rollback()

            # Update certificate status to failed
            if 'certificate' in locals():
                try:
                    certificate.processing_status = "failed" # type: ignore
                    certificate.processing_error = str(e) # type: ignore
                    db.commit()
                except Exception as update_error:
                    logger.error(f"Failed to update certificate status: {update_error}")
                    db.rollback()
            else:
                try:
                    # Use the model that was last tried, or default to GPT
                    failed_model = extraction_model_used if 'extraction_model_used' in locals() else EXTRACTION_MODEL
                    # Ensure attempts_used is defined (should be 0 if extraction never started)
                    attempts_value = attempts_used if 'attempts_used' in locals() else 1
                    failure_record = CertificateDB(
                        id=certificate_id,
                        user_id=user_id,
                        project_name=f"Processing Failed: {filename}",
                        original_filename=filename,
                        file_path=file_path,
                        file_hash=file_hash,
                        file_size=file_size,
                        extracted_text="",
                        processing_status="failed",
                        processing_error=str(e),
                        extraction_method="vision_direct",
                        parsing_method=f"{failed_model}_json",
                        extraction_quality_score=0.0,
                        extraction_attempts=attempts_value,
                        created_at=datetime.utcnow(),
                        processed_at=datetime.utcnow()
                    )
                    db.add(failure_record)
                    db.commit()
                except Exception as record_error:
                    logger.error(f"Failed to record certificate failure state: {record_error}")
                    db.rollback()

            raise
        finally:
            db.close()

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string into datetime object."""
        if not date_str:
            return None

        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d-%b-%Y',
                '%d %b %Y',
                '%d %B %Y',
                '%b %d %Y',
                '%B %d %Y',
                '%d.%m.%Y'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None
        except:
            return None

    def search_certificates(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search certificates using vector similarity.

        Args:
            query: Search query string
            user_id: User ID to filter results
            limit: Maximum number of results

        Returns:
            List of matching certificates with scores
        """
        db = SessionLocal()

        try:
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])[0]
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Get all certificate vectors for this user (focus on full_text embeddings)
            vectors = db.query(VectorDB).join(CertificateDB).filter(
                CertificateDB.user_id == user_id,
                CertificateDB.processing_status == "completed",
                VectorDB.content_type == "full_text"  # Only search full text embeddings
            ).all()

            if not vectors:
                return []

            results = []

            # If FAISS indices exist, use them, otherwise use direct cosine similarity
            for vector in vectors:
                try:
                    if vector.faiss_index_path and os.path.exists(vector.faiss_index_path): # type: ignore
                        # Use FAISS index
                        index = faiss.read_index(vector.faiss_index_path)
                        D, I = index.search(query_vector, 1)
                        if len(D[0]) > 0:
                            score = float(D[0][0])
                    else:
                        # Use direct cosine similarity
                        if vector.embedding and len(vector.embedding) == len(query_embedding): # type: ignore
                            vector_embedding = np.array(vector.embedding, dtype=np.float32)
                            # Normalize both vectors
                            faiss.normalize_L2(vector_embedding.reshape(1, -1))
                            # Calculate cosine similarity
                            score = float(np.dot(query_vector[0], vector_embedding))
                        else:
                            continue

                    results.append({
                        'certificate': vector.certificate,
                        'score': score, # type: ignore
                        'content_type': vector.content_type
                    })

                except Exception as e:
                    logger.error(f"Vector similarity calculation failed for {vector.id}: {e}")
                    continue

            # Sort by score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Certificate search failed: {e}")
            return []
        finally:
            db.close()

    def delete_certificate_with_cleanup(self, certificate_id: str, user_id: str) -> bool:
        """
        Delete a certificate and clean up associated FAISS indices.

        Args:
            certificate_id: ID of the certificate to delete
            user_id: ID of the user (for authorization)

        Returns:
            True if deletion was successful, False otherwise
        """
        db = SessionLocal()

        try:
            certificate = db.query(CertificateDB).filter(
                CertificateDB.id == certificate_id,
                CertificateDB.user_id == user_id
            ).first()

            if not certificate:
                logger.warning(f"Certificate {certificate_id} not found for user {user_id}")
                return False

            # Get all associated vectors
            vectors = db.query(VectorDB).filter(
                VectorDB.certificate_id == certificate_id
            ).all()

            # Delete FAISS index files
            for vector in vectors:
                if vector.faiss_index_path and os.path.exists(vector.faiss_index_path):
                    try:
                        os.remove(vector.faiss_index_path)
                        logger.info(f"Deleted FAISS index: {vector.faiss_index_path}")
                    except Exception as e:
                        logger.error(f"Error deleting FAISS index {vector.faiss_index_path}: {e}")

            for vector in vectors:
                db.delete(vector)

            if certificate.file_path and os.path.exists(certificate.file_path):
                try:
                    os.remove(certificate.file_path)
                    logger.info(f"Deleted certificate file: {certificate.file_path}")
                except Exception as e:
                    logger.error(f"Error deleting certificate file {certificate.file_path}: {e}")

            # Delete from database (cascade will delete vectors)
            db.delete(certificate)
            db.commit()

            logger.info(f"Successfully deleted certificate {certificate_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting certificate {certificate_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def cleanup_orphaned_faiss_files(self) -> int:
        """
        Remove FAISS index files that are not referenced in the database.

        Returns:
            Number of orphaned files deleted
        """
        if not os.path.exists(FAISS_INDEX_DIR):
            return 0

        db = SessionLocal()
        deleted_count = 0

        try:
            # Get all FAISS files on disk
            faiss_files = set(os.listdir(FAISS_INDEX_DIR))

            # Get all FAISS paths in database
            db_paths = set([
                Path(v.faiss_index_path).name
                for v in db.query(VectorDB.faiss_index_path).all()
                if v.faiss_index_path
            ])

            # Find orphaned files
            orphaned = faiss_files - db_paths

            # Remove orphaned files
            for file in orphaned:
                file_path = os.path.join(FAISS_INDEX_DIR, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Deleted orphaned FAISS file: {file}")
                except Exception as e:
                    logger.error(f"Error deleting orphaned file {file}: {e}")

            logger.info(f"Cleanup complete: {deleted_count} orphaned FAISS files deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during FAISS cleanup: {e}")
            return deleted_count
        finally:
            db.close()


# Global processor instance
certificate_processor = CertificateProcessor()
