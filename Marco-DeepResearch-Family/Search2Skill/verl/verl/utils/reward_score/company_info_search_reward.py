import re
import random
from typing import Dict, Any, Optional, List


def extract_solution(solution_str: str) -> Optional[str]:
    """Extracts the content within <answer> and </answer> tags."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def count_answer_tags(text: str) -> tuple[int, int]:
    """Counts the occurrences of <answer> and </answer> tags."""
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags


def extract_company_names(answer_text: str) -> List[str]:
    """
    Extract company names from the answer text.
    Assumes company names are listed one per line or separated by newlines.
    """
    if not answer_text:
        return []
    
    # Split by newlines and clean up
    lines = answer_text.strip().split('\n')
    company_names = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove common list markers (bullets, numbers, etc.)
        line = re.sub(r'^[-*•\d]+[\.\)]\s*', '', line)
        line = line.strip()
        
        if line:
            company_names.append(line.lower())  # Normalize to lowercase for matching
    
    return company_names


def normalize_company_name(name: str) -> str:
    """
    Normalize company name for strict matching.
    - Convert to lowercase
    - Remove all punctuation
    - Remove redundant spaces
    """
    # Convert to lowercase
    name = name.lower()
    # Remove all punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Remove redundant spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    format_score: float = 0.0,
    score: float = 1.0,
    yes_score: float = 1.0,
    no_score: float = -1.0,
    unsure_score: float = 0.0,
) -> float:
    """
    Compute the reward score for company finding task.
    
    Args:
        solution_str: The model's generated solution string.
        ground_truth: A dictionary containing ground truth information:
            - yes_list: List of company names with verify="yes"
            - no_list: List of company names with verify="no"
            - unsure_list: List of company names with verify="unsure"
        extra_info: Additional information (optional).
        format_score: Base score for proper format (default: 0.0).
        score: Maximum score for a correct answer (default: 1.0).
        yes_score: Score for matching a "yes" company (default: 1.0).
        no_score: Score for matching a "no" company (default: -1.0).
        unsure_score: Score for matching an "unsure" company (default: 0.0).
    
    Returns:
        float: The computed reward score.
    
    Scoring logic:
        - For each company mentioned in the answer:
          - If it matches a "yes" company: count towards yes_component
          - If it matches a "no" company: count towards no_component (typically negative)
          - If it matches an "unsure" company: count towards unsure_component
        - Uses strict matching after normalization (lowercase, remove punctuation, trim spaces)
        - Uses set-based lookup for O(1) matching complexity
        - Each component is calculated as: (matched_count / total_count) × component_score
        - Final score = yes_component + no_component + unsure_component
        - Allows negative scores when no/wrong companies outweigh yes companies
        - Clipped to maximum score (but allows negative)
        - Penalize incorrect format or excessive tags
    """
    extracted_answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    
    # Random sampling for logging
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print("=" * 50)
        print(f"Ground Truth Info:")
        print(f"  Yes list length: {len(ground_truth.get('yes_list', []))}")
        print(f"  No list length: {len(ground_truth.get('no_list', []))}")
        print(f"  Unsure list length: {len(ground_truth.get('unsure_list', []))}")
        print(f"Scoring config: yes={yes_score}, no={no_score}, unsure={unsure_score}")
        print(f"Solution string (first 200 chars): {solution_str[:200]}...")
        print(f"Extracted answer: {extracted_answer[:200] if extracted_answer else None}...")
    
    # Check format
    if extracted_answer is None:
        if do_print:
            print("No answer extracted. Score: 0.0")
        return 0.0
    
    # Anti-cheating: penalize excessive tags
    if open_count > 10 or close_count > 10:
        if do_print:
            print(f"Excessive answer tags detected. Score: {score / 4}")
        return score / 4
    
    # Extract company names from the answer
    extracted_companies = extract_company_names(extracted_answer)
    
    if not extracted_companies:
        if do_print:
            print("No companies extracted from answer. Score: format_score")
        return format_score
    
    # Get ground truth company lists
    yes_list = ground_truth.get("yes_list", [])
    no_list = ground_truth.get("no_list", [])
    unsure_list = ground_truth.get("unsure_list", [])
    
    if not yes_list and not no_list and not unsure_list:
        if do_print:
            print("No ground truth company lists. Score: format_score")
        return format_score
    
    # Normalize and convert to sets for O(1) lookup
    yes_set = {normalize_company_name(name) for name in yes_list}
    no_set = {normalize_company_name(name) for name in no_list}
    unsure_set = {normalize_company_name(name) for name in unsure_list}
    
    # Match extracted companies against ground truth
    matched_yes = 0
    matched_no = 0
    matched_unsure = 0
    
    for extracted_name in extracted_companies:
        normalized_name = normalize_company_name(extracted_name)
        
        # Check against yes companies (O(1) lookup)
        if normalized_name in yes_set:
            matched_yes += 1
        # Check against no companies
        elif normalized_name in no_set:
            matched_no += 1
        # Check against unsure companies
        elif normalized_name in unsure_set:
            matched_unsure += 1
    
    # Calculate normalized score for each category separately
    yes_count = len(yes_list)
    no_count = len(no_list)
    unsure_count = len(unsure_list)
    
    # Normalize each matched count by its category total, then multiply by score
    yes_component = (matched_yes / yes_count * yes_score) if yes_count > 0 else 0.0
    no_component = (matched_no / no_count * no_score) if no_count > 0 else 0.0
    unsure_component = (matched_unsure / unsure_count * unsure_score) if unsure_count > 0 else 0.0
    
    # Sum all components to get final score
    final_score = yes_component + no_component + unsure_component
    
    # Clip to maximum score (allow negative scores for penalties)
    final_score = min(score, final_score)
    
    if do_print:
        print(f"Extracted companies: {len(extracted_companies)}")
        print(f"Matched yes: {matched_yes}/{yes_count} (×{yes_score}), Matched no: {matched_no}/{no_count} (×{no_score}), Matched unsure: {matched_unsure}/{unsure_count} (×{unsure_score})")
        print(f"Yes component: {matched_yes}/{yes_count} × {yes_score} = {yes_component:.4f}")
        print(f"No component: {matched_no}/{no_count} × {no_score} = {no_component:.4f}")
        print(f"Unsure component: {matched_unsure}/{unsure_count} × {unsure_score} = {unsure_component:.4f}")
        print(f"Sum before clipping: {yes_component + no_component + unsure_component:.4f}")
        print(f"Final score (clipped to max={score}): {final_score:.4f}")
        print("=" * 50)
    
    return final_score


