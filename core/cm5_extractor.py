"""
PQ/Eligibility Criteria Extractor
Uses OpenAI GPT-4o to intelligently extract and structure criteria from RFP documents
"""

from openai import OpenAI
import openai
import json
from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)


class CM5PQExtractor:
    """Extracts PQ and Eligibility criteria using LLM"""

    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    def extract_criteria(self, text_content: str) -> Dict:
        """
        Extract PQ and Eligibility criteria from RFP text

        Args:
            text_content: Full text of the RFP document

        Returns:
            Dictionary with structured extraction results
        """
        # Validate input text
        if not text_content or text_content.strip() == "":
            logger.error("Empty text content provided for extraction")
            return {
                'pq_criteria': [],
                'eligibility_criteria': [],
                'sections': [],
                'error': 'Empty document text - possibly scanned PDF without OCR',
                'raw_response': ''
            }

        if len(text_content) < 500:
            logger.warning(f"Very short text content ({len(text_content)} chars) - may not contain criteria")

        try:
            # Create the extraction prompt
            prompt = self._create_extraction_prompt(text_content)

            # Call OpenAI API with increased token limit
            logger.info(f"Calling GPT-4o for criteria extraction (text length: {len(text_content)} chars)")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document extraction specialist. Your job is to extract text VERBATIM (word-for-word) from documents. You NEVER paraphrase, summarize, or rewrite. You copy the exact text as it appears in the source document. You are highly accurate and detail-oriented."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=16000,
                response_format={"type": "json_object"}
            )

            # Parse response
            result_text = response.choices[0].message.content
            logger.debug(f"Received GPT response: {len(result_text)} chars")

            try:
                extraction_result = json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from GPT: {result_text[:500]}...", exc_info=True)
                return {
                    'pq_criteria': [],
                    'eligibility_criteria': [],
                    'sections': [],
                    'error': f'Invalid JSON response from GPT: {str(e)}',
                    'raw_response': result_text
                }

            # Validate result
            pq_count = len(extraction_result.get('pq_criteria', []))
            eligibility_count = len(extraction_result.get('eligibility_criteria', []))

            if pq_count + eligibility_count == 0:
                logger.warning("GPT returned zero criteria - document may not contain PQ/Eligibility sections")

            logger.info(f"Extraction successful: {pq_count} PQ criteria, {eligibility_count} eligibility criteria")

            # Post-process and validate
            extraction_result = self._post_process_extraction(extraction_result)

            # Add raw response for transparency
            extraction_result['raw_response'] = result_text

            return extraction_result

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return {
                'pq_criteria': [],
                'eligibility_criteria': [],
                'sections': [],
                'error': f'OpenAI API error: {str(e)}',
                'raw_response': ''
            }
        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}", exc_info=True)
            return {
                'pq_criteria': [],
                'eligibility_criteria': [],
                'sections': [],
                'error': f'OpenAI API timeout: {str(e)}',
                'raw_response': ''
            }
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
            return {
                'pq_criteria': [],
                'eligibility_criteria': [],
                'sections': [],
                'error': f'Unexpected error: {str(e)}',
                'raw_response': ''
            }

    def _create_extraction_prompt(self, text_content: str) -> str:
        """Create detailed extraction prompt"""

        # Truncate if too long (keep first 80k chars for better coverage)
        original_length = len(text_content)
        if original_length > 80000:
            logger.warning(f"Document text truncated from {original_length} to 80000 chars - may lose content from later sections")
            text_content = text_content[:80000] + "\n\n[Document truncated for processing]"

        prompt = f"""
You are extracting PQ (Pre-Qualification) and Eligibility criteria from an RFP document.

**CRITICAL INSTRUCTION: Extract criteria VERBATIM (word-for-word) from the document. Do NOT paraphrase, summarize, or rewrite. Copy the EXACT text as it appears.**

**What to Extract:**

1. **PQ Criteria** - Look for sections labeled as:
   - "Pre-Qualification Criteria"
   - "PQ Requirements"
   - "Qualifying Criteria"
   - "Technical Qualification"
   - "Financial Qualification"
   - Similar headings

2. **Eligibility Criteria** - Look for sections labeled as:
   - "Eligibility Criteria"
   - "Eligibility Requirements"
   - "General Eligibility"
   - "Bidder Eligibility"
   - Similar headings

**For EACH criterion found:**
- **title**: Extract the criterion heading/number EXACTLY as written (e.g., "2.1 Financial Capacity", "Eligibility Criterion 1")
- **description**: Copy the COMPLETE TEXT of the requirement VERBATIM. Include all details, numbers, conditions exactly as written
- **value**: If a specific value/threshold is mentioned (amount, years, percentage), copy it EXACTLY
- **type**: Classify as one of: financial, technical, experience, registration, certification, legal, other
- **is_mandatory**: Determine if mandatory (true) or optional (false) based on keywords like "must", "shall", "should"
- **page_reference**: Note the page number if visible (e.g., "Page 5")

**IMPORTANT RULES:**
1. Extract EVERY criterion listed under PQ and Eligibility sections
2. Use EXACT wording from document - do not paraphrase or summarize
3. Include complete sentences and all details for each criterion
4. If a criterion has sub-points (a, b, c), extract each sub-point separately
5. Preserve numbering schemes (e.g., "2.1.1", "Clause 3.2")
6. Copy any tables, lists, or structured data verbatim

**Return JSON format:**
{{
    "pq_criteria": [
        {{
            "type": "financial",
            "title": "[EXACT heading from document]",
            "description": "[COMPLETE verbatim text of the requirement with all details]",
            "value": "[Exact value if specified]",
            "is_mandatory": true,
            "page_reference": "Page X"
        }}
    ],
    "eligibility_criteria": [
        {{
            "type": "registration",
            "title": "[EXACT heading from document]",
            "description": "[COMPLETE verbatim text of the requirement with all details]",
            "value": "[Exact value if specified]",
            "is_mandatory": true,
            "page_reference": "Page Y"
        }}
    ],
    "sections": [
        {{
            "title": "[Exact section heading]",
            "content": "[Complete verbatim section text]",
            "type": "pq",
            "pages": ["X", "Y"]
        }}
    ],
    "summary": {{
        "total_pq_criteria": 0,
        "total_eligibility_criteria": 0,
        "mandatory_count": 0,
        "optional_count": 0
    }}
}}

**RFP Document Text:**

{text_content}

**REMEMBER: Extract VERBATIM - use exact text from document, do not paraphrase. Return valid JSON only.**
"""
        return prompt

    def _post_process_extraction(self, extraction: Dict) -> Dict:
        """Post-process and validate extraction results"""

        # Ensure required keys exist
        if 'pq_criteria' not in extraction:
            extraction['pq_criteria'] = []
        if 'eligibility_criteria' not in extraction:
            extraction['eligibility_criteria'] = []
        if 'sections' not in extraction:
            extraction['sections'] = []

        # Validate and clean each criterion
        extraction['pq_criteria'] = [
            self._validate_criterion(c) for c in extraction['pq_criteria']
        ]
        extraction['eligibility_criteria'] = [
            self._validate_criterion(c) for c in extraction['eligibility_criteria']
        ]

        # Add summary if not present
        if 'summary' not in extraction:
            extraction['summary'] = self._generate_summary(extraction)

        return extraction

    def _validate_criterion(self, criterion: Dict) -> Dict:
        """Validate and clean a single criterion"""
        # Ensure required fields
        if 'type' not in criterion:
            criterion['type'] = 'general'
        if 'title' not in criterion:
            criterion['title'] = 'Untitled Criterion'
        if 'description' not in criterion:
            criterion['description'] = ''
        if 'value' not in criterion:
            criterion['value'] = ''
        if 'is_mandatory' not in criterion:
            criterion['is_mandatory'] = True
        if 'page_reference' not in criterion:
            criterion['page_reference'] = ''

        # Clean text fields
        criterion['title'] = str(criterion['title']).strip()
        criterion['description'] = str(criterion['description']).strip()
        criterion['value'] = str(criterion['value']).strip()

        return criterion

    def _generate_summary(self, extraction: Dict) -> Dict:
        """Generate summary statistics"""
        pq_count = len(extraction.get('pq_criteria', []))
        eligibility_count = len(extraction.get('eligibility_criteria', []))

        all_criteria = extraction.get('pq_criteria', []) + extraction.get('eligibility_criteria', [])
        mandatory_count = sum(1 for c in all_criteria if c.get('is_mandatory', True))
        optional_count = len(all_criteria) - mandatory_count

        return {
            'total_pq_criteria': pq_count,
            'total_eligibility_criteria': eligibility_count,
            'mandatory_count': mandatory_count,
            'optional_count': optional_count,
            'total_criteria': len(all_criteria)
        }

    def extract_specific_section(self, text_content: str, section_name: str) -> str:
        """
        Extract a specific section from the document

        Args:
            text_content: Full document text
            section_name: Name of section to extract

        Returns:
            Extracted section content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document section extractor. Extract the requested section accurately."
                    },
                    {
                        "role": "user",
                        "content": f"Extract the '{section_name}' section from the following document:\n\n{text_content[:20000]}"
                    }
                ],
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error extracting section: {str(e)}"

    def validate_criteria_completeness(self, extraction: Dict) -> Dict:
        """
        Validate if extraction is complete and comprehensive

        Args:
            extraction: Extraction result dictionary

        Returns:
            Validation report
        """
        issues = []
        warnings = []

        # Check if criteria were found
        if not extraction.get('pq_criteria') and not extraction.get('eligibility_criteria'):
            issues.append("No criteria found in document")

        # Check for missing values
        all_criteria = extraction.get('pq_criteria', []) + extraction.get('eligibility_criteria', [])
        missing_values = [c for c in all_criteria if not c.get('value')]
        if missing_values:
            warnings.append(f"{len(missing_values)} criteria have no specific values")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'completeness_score': self._calculate_completeness(extraction)
        }

    def _calculate_completeness(self, extraction: Dict) -> float:
        """Calculate completeness score (0-1)"""
        score = 0.0

        # Has criteria
        if extraction.get('pq_criteria') or extraction.get('eligibility_criteria'):
            score += 0.4

        # Has sections
        if extraction.get('sections'):
            score += 0.2

        # Has summary
        if extraction.get('summary'):
            score += 0.2

        # Has page references
        all_criteria = extraction.get('pq_criteria', []) + extraction.get('eligibility_criteria', [])
        if all_criteria:
            with_refs = sum(1 for c in all_criteria if c.get('page_reference'))
            score += 0.2 * (with_refs / len(all_criteria))

        return min(1.0, score)
