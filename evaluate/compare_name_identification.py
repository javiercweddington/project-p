"""
Compare name identification speed and accuracy between three methods:
1. Regex-based (pattern matching)
2. GLiNER (pre-trained NER model)
3. Local LLM (Qwen via OpenAI-compatible API at localhost:8000)

This script extracts text from a single PDF document and compares timing and accuracy.

Usage:
    python compare_name_identification.py <zip_file> <pdf_entry_path>
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Add parent directory to path to import catalog module
sys.path.insert(0, str(Path(__file__).parent))

from acquire.catalog import (
    extract_pdf_text_from_zip,
    extract_entities_from_text_regex,
    _extract_entities_with_gliner,
    _get_gliner_model,
    EntityHit,
)

# Local LLM API settings
LLM_API_BASE = "http://localhost:8000/v1"
LLM_MODEL = "qwen27b"


def call_llm_for_entities(text: str, prompt: str) -> List[Dict[str, Any]]:
    """Call the local LLM API to extract entities from text."""
    if not HAS_OPENAI:
        print("Warning: openai package not available.")
        return []
    
    client = OpenAI(base_url=LLM_API_BASE, api_key="not-needed")
    
    # Truncate text if too long
    max_length = 4000
    if len(text) > max_length:
        text = text[:max_length] + "\n... (truncated)"
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content
        print(f"  LLM raw response (first 500 chars): {result_text[:500]}...")
        print(f"  LLM raw response (last 300 chars): ...{result_text[-300:]}")
        
        # The LLM may output thinking/reasoning before the actual JSON answer.
        # Strategy: find the LAST JSON array or object in the response,
        # as that's typically the final answer after reasoning.
        
        import re as re_module
        
        # Try to find JSON arrays - look for the last one
        array_matches = list(re_module.finditer(r'\[[^\]]*\]', result_text, re_module.DOTALL))
        for match in reversed(array_matches):
            json_str = match.group()
            try:
                result = json.loads(json_str)
                if isinstance(result, list) and len(result) > 0:
                    # Validate it looks like entity data
                    if isinstance(result[0], dict) and 'type' in result[0] and 'value' in result[0]:
                        print(f"  Parsed {len(result)} entities from array")
                        return result
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON objects with 'entities' key - look for the last one
        # Simple heuristic: find {...} blocks
        depth = 0
        start_idx = -1
        object_candidates = []
        for i, ch in enumerate(result_text):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    object_candidates.append(result_text[start_idx:i+1])
        
        for json_str in reversed(object_candidates):
            try:
                result = json.loads(json_str)
                if isinstance(result, dict) and 'entities' in result:
                    entities = result['entities']
                    if isinstance(entities, list):
                        print(f"  Parsed {len(entities)} entities from object")
                        return entities
            except json.JSONDecodeError:
                continue
        
        print(f"Could not parse JSON from LLM response")
        return []
            
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []


ENTITY_EXTRACTION_PROMPT = """Extract all person names and company names from the text below.

Output format - a JSON array:
[{"type":"person","value":"NAME"}, {"type":"company","value":"COMPANY"}]

Rules:
- Only extract real names from the text
- Persons = first name + last name  
- Companies = business/organization names
- Skip generic terms like "owing party", "owed party"
- Return empty array [] if no names found
"""


def get_entity_values(entities: List[EntityHit], entity_type: str) -> Set[str]:
    """Extract unique entity values of a given type, normalized to lowercase."""
    return {e.value.lower() for e in entities if e.entity_type == entity_type}


def llm_entities_to_hits(llm_result: List[Dict[str, Any]], source: str) -> List[EntityHit]:
    """Convert LLM entity results to EntityHit objects."""
    hits = []
    for entity in llm_result:
        entity_type = entity.get('type', '').lower()
        value = entity.get('value', '').strip()
        if value:
            if entity_type in ('organization', 'org'):
                entity_type = 'company'
            hits.append(EntityHit(
                entity_type=entity_type,
                value=value,
                source=source,
                confidence=0.9
            ))
    return hits


def compute_accuracy(ground_truth: Set[str], predicted: Set[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
    if not ground_truth and not predicted:
        return 0.0, 0.0, 0.0
    
    true_positives = len(predicted & ground_truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def print_overlap_analysis(entities_a: Set[str], entities_b: Set[str], name_a: str, name_b: str, title: str):
    """Print overlap analysis between two entity sets."""
    print(f"\n{title} ({name_a} vs {name_b}):")
    
    both = entities_a & entities_b
    only_a = entities_a - entities_b
    only_b = entities_b - entities_b
    
    # Check substring matches
    for a in list(only_a):
        for b in entities_b:
            if a in b or b in a:
                both.add(a)
                only_a.discard(a)
                break
    
    for b in list(entities_b - entities_a):
        for a in entities_a:
            if b in a or a in b:
                both.add(b)
                break
    
    only_b = entities_b - both
    only_a = entities_a - both
    
    print(f"  Found by both:      {len(both)}")
    print(f"  Found by {name_a} only: {len(only_a)}")
    print(f"  Found by {name_b} only: {len(only_b)}")
    
    if both:
        print(f"  Common entities:")
        for e in sorted(both):
            print(f"    ✓ {e}")
    
    if only_a:
        print(f"  Only in {name_a}:")
        for e in sorted(only_a):
            print(f"    - {e}")
    
    if only_b:
        print(f"  Only in {name_b}:")
        for e in sorted(only_b):
            print(f"    - {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_name_identification.py <zip_file> <pdf_entry_path>")
        sys.exit(1)
    
    zip_path = Path(sys.argv[1])
    pdf_entry = sys.argv[2]
    
    if not zip_path.exists():
        print(f"Error: Zip file not found: {zip_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Name Identification Comparison: Regex vs GLiNER vs Local LLM")
    print("=" * 80)
    print(f"\nZip file: {zip_path}")
    print(f"PDF entry: {pdf_entry}")
    print()
    
    # Step 1: Extract text from PDF
    print("-" * 80)
    print("Step 1: Extracting text from PDF...")
    print("-" * 80)
    
    start = time.perf_counter()
    text = extract_pdf_text_from_zip(zip_path, pdf_entry, max_pages=5)
    extract_time = time.perf_counter() - start
    
    if not text:
        print("Error: Could not extract text from PDF")
        sys.exit(1)
    
    print(f"Text extraction time: {extract_time:.3f}s")
    print(f"Extracted {len(text)} characters")
    print(f"Preview: {text[:300]}...")
    print()
    
    # Step 2: Regex-based entity detection
    print("-" * 80)
    print("Step 2: Regex-based entity detection...")
    print("-" * 80)
    
    start = time.perf_counter()
    regex_entities = extract_entities_from_text_regex(text, pdf_entry)
    regex_time = time.perf_counter() - start
    
    regex_persons = get_entity_values(regex_entities, 'person')
    regex_companies = get_entity_values(regex_entities, 'company')
    
    print(f"Regex detection time: {regex_time:.4f}s")
    print(f"Found {len(regex_entities)} entities total")
    print(f"  Persons: {len(regex_persons)}")
    print(f"  Companies: {len(regex_companies)}")
    
    if regex_persons:
        print("  Persons:")
        for p in sorted(regex_persons):
            print(f"    - {p}")
    if regex_companies:
        print("  Companies:")
        for c in sorted(regex_companies):
            print(f"    - {c}")
    print()
    
    # Step 3: GLiNER-based entity detection
    print("-" * 80)
    print("Step 3: GLiNER entity detection...")
    print("-" * 80)
    
    model = _get_gliner_model()
    if model is None:
        print("GLiNER model not available, skipping...")
        gliner_entities = []
        gliner_time = 0.0
    else:
        start = time.perf_counter()
        gliner_entities = _extract_entities_with_gliner(text, pdf_entry)
        gliner_time = time.perf_counter() - start
    
    gliner_persons = get_entity_values(gliner_entities, 'person')
    gliner_companies = get_entity_values(gliner_entities, 'company')
    
    if gliner_entities:
        print(f"GLiNER detection time: {gliner_time:.4f}s")
        print(f"Found {len(gliner_entities)} entities total")
        print(f"  Persons: {len(gliner_persons)}")
        print(f"  Companies: {len(gliner_companies)}")
        
        if gliner_persons:
            print("  Persons:")
            for p in sorted(gliner_persons):
                print(f"    - {p}")
        if gliner_companies:
            print("  Companies:")
            for c in sorted(gliner_companies):
                print(f"    - {c}")
    print()
    
    # Step 4: LLM-based entity detection
    print("-" * 80)
    print("Step 4: Local LLM entity detection...")
    print("-" * 80)
    
    start = time.perf_counter()
    llm_result = call_llm_for_entities(text, ENTITY_EXTRACTION_PROMPT)
    llm_time = time.perf_counter() - start
    
    llm_entities = llm_entities_to_hits(llm_result, pdf_entry)
    llm_persons = get_entity_values(llm_entities, 'person')
    llm_companies = get_entity_values(llm_entities, 'company')
    
    print(f"LLM detection time: {llm_time:.4f}s")
    print(f"Found {len(llm_entities)} entities total")
    print(f"  Persons: {len(llm_persons)}")
    print(f"  Companies: {len(llm_companies)}")
    
    if llm_persons:
        print("  Persons:")
        for p in sorted(llm_persons):
            print(f"    - {p}")
    if llm_companies:
        print("  Companies:")
        for c in sorted(llm_companies):
            print(f"    - {c}")
    print()
    
    # Step 5: Comparison
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Timing comparison
    print("TIMING:")
    print(f"  Regex:   {regex_time:.4f}s")
    if gliner_entities:
        print(f"  GLiNER:  {gliner_time:.4f}s")
    print(f"  LLM:     {llm_time:.4f}s")
    
    if regex_time > 0:
        print(f"  GLiNER is {gliner_time/regex_time:.1f}x vs Regex" if gliner_entities else "  GLiNER: N/A")
        print(f"  LLM is {llm_time/regex_time:.1f}x vs Regex")
    print()
    
    # Entity count comparison
    print("ENTITY COUNT:")
    print(f"  {'Method':<15} {'Persons':<10} {'Companies':<10} {'Total':<10}")
    print(f"  {'-'*45}")
    print(f"  {'Regex':<15} {len(regex_persons):<10} {len(regex_companies):<10} {len(regex_entities):<10}")
    if gliner_entities:
        print(f"  {'GLiNER':<15} {len(gliner_persons):<10} {len(gliner_companies):<10} {len(gliner_entities):<10}")
    print(f"  {'LLM':<15} {len(llm_persons):<10} {len(llm_companies):<10} {len(llm_entities):<10}")
    print()
    
    # Overlap analysis for persons
    print("=" * 80)
    print("PERSONS OVERLAP ANALYSIS")
    print("=" * 80)
    
    all_person_sets = [
        ("Regex", regex_persons),
        ("LLM", llm_persons),
    ]
    if gliner_entities:
        all_person_sets.insert(1, ("GLiNER", gliner_persons))
    
    for i in range(len(all_person_sets)):
        for j in range(i + 1, len(all_person_sets)):
            name_a, entities_a = all_person_sets[i]
            name_b, entities_b = all_person_sets[j]
            print_overlap_analysis(entities_a, entities_b, name_a, name_b, 
                                  f"Overlap")
            print()
    
    # Overlap analysis for companies
    print("=" * 80)
    print("COMPANIES OVERLAP ANALYSIS")
    print("=" * 80)
    
    all_company_sets = [
        ("Regex", regex_companies),
        ("LLM", llm_companies),
    ]
    if gliner_entities:
        all_company_sets.insert(1, ("GLiNER", gliner_companies))
    
    for i in range(len(all_company_sets)):
        for j in range(i + 1, len(all_company_sets)):
            name_a, entities_a = all_company_sets[i]
            name_b, entities_b = all_company_sets[j]
            print_overlap_analysis(entities_a, entities_b, name_a, name_b,
                                  f"Overlap")
            print()
    
    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Document: {pdf_entry}")
    print(f"Text length: {len(text)} characters")
    print()
    print(f"{'Metric':<25} {'Regex':<15}", end='')
    if gliner_entities:
        print(f" {'GLiNER':<15}", end='')
    print(f" {'LLM':<15}")
    print(f"{'-'*(25 + 15 + (15 if gliner_entities else 0) + 15)}")
    
    row = f"{'Detection time (s)':<25} {regex_time:<15.4f}"
    if gliner_entities:
        row += f" {gliner_time:<15.4f}"
    row += f" {llm_time:<15.4f}"
    print(row)
    
    row = f"{'Persons found':<25} {len(regex_persons):<15}"
    if gliner_entities:
        row += f" {len(gliner_persons):<15}"
    row += f" {len(llm_persons):<15}"
    print(row)
    
    row = f"{'Companies found':<25} {len(regex_companies):<15}"
    if gliner_entities:
        row += f" {len(gliner_companies):<15}"
    row += f" {len(llm_companies):<15}"
    print(row)
    
    row = f"{'Total entities':<25} {len(regex_entities):<15}"
    if gliner_entities:
        row += f" {len(gliner_entities):<15}"
    row += f" {len(llm_entities):<15}"
    print(row)
    
    print()
    print("CONCLUSION:")
    print(f"  - Regex is the fastest method ({regex_time:.4f}s)")
    if llm_time > regex_time:
        print(f"  - LLM is {llm_time/regex_time:.1f}x slower than Regex")
    if gliner_entities and gliner_time > regex_time:
        print(f"  - GLiNER is {gliner_time/regex_time:.1f}x slower than Regex")
    
    # Find unique entities
    all_persons = regex_persons | llm_persons
    if gliner_entities:
        all_persons |= gliner_persons
    
    if len(all_persons) > len(regex_persons):
        print(f"  - LLM/GLiNER found persons that Regex missed")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()