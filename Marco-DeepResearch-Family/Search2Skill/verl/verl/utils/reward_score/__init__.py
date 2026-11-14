# from . import gsm8k, math, prime_math, prime_code

import asyncio
from concurrent.futures import ThreadPoolExecutor

from verl.utils.import_utils import deprecated

import fnmatch

MULTI_HOP_QA_PATTERNS = [
    "BDInfoSearch", "searchR1_webdancer", "search_rtp_compare", "beerqa",
    "sailorfog-QA", "WebDancer", "WebShaper", "xbench-DeepSearch*", 
    "DeepSearchQA", "gaia", "musique_filtered", 
    "qa-expert-multi-hop-qa-V1.0_filtered", "deepdive_qa_rl*",
    "BrowseComp-*", "2WikiMultihopQA_*", "MuSiQue_*", "BeerQA_*"
]

def is_multi_hop_qa_source(data_source):
    """检查是否为多跳问答数据源"""
    return any(fnmatch.fnmatch(data_source, pattern) for pattern in MULTI_HOP_QA_PATTERNS)

def extract_after_last_substring(original_string, substring):
    # 查找子串在原始字符串中最后一次出现的位置
    last_occurrence = original_string.rfind(substring)
    
    # 如果找到子串
    if last_occurrence != -1:
        # 计算子串结束的位置
        start_index = last_occurrence + len(substring)
        # 返回子串之后的所有内容
        return original_string[start_index:]
    else:
        # 如果没有找到子串，返回原始字符串
        return original_string

def _chk_format_score(s,data_source):
    s = s.strip()
    if data_source.endswith('_think') and not s.startswith('<think>'):
        return -1.0
    closing_tag_count = s.count('</think>')
    if closing_tag_count == 1:
        return 1.0
    return -1.0

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    format_score=_chk_format_score(solution_str,data_source)
    real_solution=extract_after_last_substring(solution_str,"</think>")

    # print("=============================================")
    # print(f"Full Solution:\n {solution_str}")
    # print(f"\nformat score:{format_score}")

    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco", "codeforces_think", "codeforces_dapo"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
            if data_source.endswith('_dapo'):
                res={
                    'score':float(res)+format_score,
                    'acc':float(res),
                    'format':format_score
                }
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
        # "qa-expert-multi-hop-qa-V1.0_filtered"
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source in ["company_info_search"]:
        from . import company_info_search_reward

        res = company_info_search_reward.compute_score(solution_str, ground_truth, extra_info, yes_score=1.0, no_score=0.0, unsure_score=0.0)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    # print(f"ground truth: {ground_truth}")
    # print(f"\naccuracy score:{res}")
    # print("=============================================")
    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])+format_score

@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )

__all__ = ["default_compute_score"]
