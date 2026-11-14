
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        futures = []
        prompt_str_list = []
        response_str_list = []
        ground_truth_list = []
        data_source_list = []
        extra_info_list = []
        valid_response_length_list = []

        with ProcessPoolExecutor() as executor:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_length_list.append(valid_response_length)
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                prompt_str_list.append(prompt_str)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                response_str_list.append(response_str)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                ground_truth_list.append(ground_truth)

                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                data_source_list.append(data_source)

                # 新增代码：获取extra_info和num_turns
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores

                # Merge agent loop runtime fields into extra_info so reward function
                # can use the real structured data instead of text-based inference.
                # NOTE: conversation_messages is intentionally excluded — it can be
                # very large (full multi-turn with tool responses) and risks OOM /
                # pickle failures in ProcessPoolExecutor.  The reward function already
                # parses assistant text from solution_str when messages are absent.
                _agent_loop_fields = (
                    "actions", "search_queries",
                    "search_count", "page_visit_count", "page_visit_success_count",
                    "page_visit_failed_count", "page_visit_forbidden_count",
                    "generated_skills", "used_explore", "explore_turns",
                    "final_answer", "explore_token_length",
                    "agent_loop_api_calls_total", "agent_loop_api_calls_success",
                    "agent_loop_api_calls_failed", "agent_loop_api_by_profile",
                    "skill_reuse_eval_results", "skill_reuse_eval_meta",
                    "skill_quality_eval_results", "skill_quality_eval_meta",
                )
                for _field in _agent_loop_fields:
                    if _field in data_item.non_tensor_batch:
                        val = data_item.non_tensor_batch[_field]
                        if val is not None:
                            extra_info[_field] = val

                extra_info_list.append(extra_info)

                # 提交compute_score任务
                future = executor.submit(
                    self.compute_score,
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                futures.append(future)

        # 收集所有compute_score的结果
        results = [future.result() for future in futures]

        for i in range(len(data)):
            result = results[i]
            data_source = data_source_list[i]
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            reward_tensor[i, valid_response_length_list[i] - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str_list[i])
                print("[response]", response_str_list[i])
                print("[ground_truth]", ground_truth_list[i])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor