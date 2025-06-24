import json
import os
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import cast

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager.self_reward import SelfRewardBase
from verl.workers.reward_manager.self_reward import decode_batch
from verl.workers.reward_manager.self_reward import encode_chat
from verl.workers.reward_manager.self_reward import to_chat_format

from legacy.integration_numeric import FormalStatus
from legacy.integration_numeric import compute_score
from legacy.integration_utils import extract_candidate_solution
from legacy.integration_utils import extract_integral
from legacy.llm_judge_utils import JudgeStatus
from legacy.llm_judge_utils import extract_judge_score

SYSTEM_PROMPT = "You are an expert at mathematical differentiation."

JUDGE_PROMPT = (
    "Please check if the following is a valid function: {response}. If it is,"
    " differentiate it and determine if it is functionally equal to {integrand}. Output"
    " <JUDGE_SCORE>1</JUDGE_SCORE> if they are equal. Output"
    " <JUDGE_SCORE>0</JUDGE_SCORE> if they are not equal or if it is not a valid"
    " function. Ignore constants of integration."
)


class Score(NamedTuple):
    judge_result: float | int
    formal_result: float | int
    judge_status: JudgeStatus
    formal_status: FormalStatus
    agent_response: str
    judge_response: str


@dataclass
class ResultsLogger:
    logs_dir: str
    logs_prefix: str

    def log_results(
        self,
        extra_infos: List[Dict[str, Any]],
        reward_models: List[Dict[str, Any]],
        results: List[Score],
        eval_step: int,
    ):
        data = []
        for extra_info, reward_model, result in zip(
            extra_infos, reward_models, results
        ):
            # TODO: It assumes non nested dicts for extra info ?
            datum = extra_info.copy() | reward_model.copy()
            datum["agent_response"] = result.agent_response
            datum["judge_response"] = result.judge_response
            datum["judge_status"] = result.judge_status.name
            datum["formal_status"] = result.formal_status.name
            datum["judge_result"] = result.judge_result
            datum["formal_result"] = result.formal_result
            data.append(datum)

        file_name = f"{self.logs_prefix}_{eval_step}.json"
        with open(os.path.join(self.logs_dir, file_name), "w") as file:
            json.dump(data, file)


class SelfRewardManager(SelfRewardBase):
    """Reward manager that uses the actor model itself to compute rewards.

    This manager requires a reference to the actor worker group to compute rewards
    using the same model that generates sequences.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        val_logger: ResultsLogger | None = None,
        judge_sampling_params: Dict[str, Any] = {},
    ) -> None:
        self.tokenizer = tokenizer
        self.val_logger = val_logger
        self.judge_sampling_params = judge_sampling_params

        # HACK: I need to keep track at which step I'm for the logs.
        # Unless I modify more our fork of verl, I don't know how to
        # retrieve at which step I'm. RewardManager should be only executed
        # by one actor. No concurrency issues are taking into account yet.
        if self.val_logger:
            self._eval_step = 0
        else:
            self._eval_step = None

    def set_actor_rollout_wg(self, actor_rollout_wg):
        """Set the reference to the actor rollout worker group.

        This must be called before using the reward manager to compute rewards.
        """
        self.actor_rollout_wg = cast(ActorRolloutRefWorker, actor_rollout_wg)

    def llm_judge(self, agent_answers: List[str], targets: List[str]):
        # Hack so that we don't pass to SGLANG queries with empty inputs.
        judge_queries = []
        for ans, target in zip(agent_answers, targets):
            if ans:
                prompt = JUDGE_PROMPT.format(response=ans, integrand=target)
                chat = to_chat_format(SYSTEM_PROMPT, prompt)
                query = encode_chat(chat, self.tokenizer, 10000, "error")
                judge_queries.append(query)
            else:
                # HACK: Verl generate_sequences does not allow variable batch
                # size. So I cannot ignore the responses without inputs.
                prompt = "Stop answering inmmediatly"
                chat = to_chat_format(SYSTEM_PROMPT, prompt)
                query = encode_chat(chat, self.tokenizer, 10000, "error")
                judge_queries.append(query)

        judge_queries = DataProto.from_single_dict(collate_fn(judge_queries))
        judge_queries.meta_info.update({"sampling_params": self.judge_sampling_params})
        judge_batch_ouput = self.actor_rollout_wg.generate_sequences(judge_queries)

        judge_responses_partial = decode_batch(judge_batch_ouput, self.tokenizer)
        assert len(agent_answers) == len(judge_responses_partial)
        judge_responses = []

        for i, ans in enumerate(agent_answers):
            if ans is not None:
                judge_responses.append(judge_responses_partial[i])
            else:
                judge_responses.append(None)

        return judge_responses

    def verify_batch(self, data: DataProto):
        targets: List[str] = [
            extract_integral(item.non_tensor_batch["reward_model"]["ground_truth"])
            for item in data
        ]
        full_responses = decode_batch(data, self.tokenizer)

        # LLM Judge
        agent_answers = [
            extract_candidate_solution(response) for response in full_responses
        ]
        judge_responses = self.llm_judge(agent_answers, targets)
        assert len(judge_responses) == len(full_responses)

        scores: List[Score] = []
        for response, agent_answer, judge_response, target in zip(
            full_responses, agent_answers, judge_responses, targets
        ):
            formal_result, formal_status = compute_score(response, target)
            if agent_answer:
                judge_result, judge_status = extract_judge_score(judge_response)
                judge_result += 0.05  # Hardcoded in the CustomTinyZero repo.
            else:
                judge_result = 0.0
                judge_status = JudgeStatus.NO_INP_IN_TAGS

            score = Score(
                judge_result=judge_result,
                formal_result=formal_result,
                judge_status=judge_status,
                formal_status=formal_status,
                agent_response=response,
                judge_response=judge_response,
            )
            scores.append(score)

        return scores, judge_responses

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Almost the same code as from BatchRewardManager from Verl.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: Dict[str, List[int] | List[float]] = {
            f"JUDGE/{status_code.name}": [0.0] * len(data)
            for status_code in JudgeStatus
        }

        for status in FormalStatus:
            reward_extra_info[f"FORMAL/{status.name}"] = [0] * len(data)

        reward_extra_info["FORMAL/REWARD"] = [0.0] * len(data)
        # HACK: Verl requires all extra info tensors to be of same
        # lenght as the size of the batch. To not mess more with the fork
        # we instead use nanmean, and nansum, which computes those
        # collectives ignoring the nans.
        for status in JudgeStatus:
            reward_extra_info[f"LENGTH/AGENT_RESP/{status.name}"] = [np.nan] * len(data)
            reward_extra_info[f"LENGTH/JUDGE/{status.name}"] = [np.nan] * len(data)

        for status in FormalStatus:
            reward_extra_info[f"LENGTH/FORMAL_AGENT_RESP/{status.name}"] = [
                np.nan
            ] * len(data)

        # Initialize confusion matrix metrics
        for metric in ["TP", "FP", "TN", "FN"]:
            reward_extra_info[f"CONFUSION_MATRIX/{metric}"] = [0] * len(data)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        scores, judge_responses = self.verify_batch(data)

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            judge_length = len(judge_responses[i]) if judge_responses[i] else 0
            judge_result = scores[i].judge_result
            judge_status = scores[i].judge_status

            reward_tensor[i, length - 1] = judge_result
            reward_extra_info[f"JUDGE/{judge_status.name}"][i] = 1
            reward_extra_info[f"LENGTH/AGENT_RESP/{judge_status.name}"][i] = length
            reward_extra_info[f"LENGTH/JUDGE/{judge_status.name}"][i] = judge_length

            # Formal Verifier
            formal_status = scores[i].formal_status
            formal_result = scores[i].formal_result
            reward_extra_info["FORMAL/REWARD"][i] = formal_result
            reward_extra_info[f"FORMAL/{formal_status.name}"][i] = 1
            reward_extra_info[f"LENGTH/FORMAL_AGENT_RESP/{formal_status.name}"][
                i
            ] = length

            # Compute confusion matrix.

            if formal_result == FormalStatus.OK and judge_result == JudgeStatus.OK:
                reward_extra_info["CONFUSION_MATRIX/TP"][i] = 1

            if formal_result == FormalStatus.NO_OK and judge_result == JudgeStatus.OK:
                reward_extra_info["CONFUSION_MATRIX/FP"][i] = 1

            if (
                formal_result == FormalStatus.NO_OK
                and judge_result == JudgeStatus.NO_OK
            ):
                reward_extra_info["CONFUSION_MATRIX/TN"][i] = 1

            if formal_result == FormalStatus.OK and judge_result == JudgeStatus.NO_OK:
                reward_extra_info["CONFUSION_MATRIX/FN"][i] = 1

        if self.val_logger:
            self.val_logger.log_results(
                data.non_tensor_batch["extra_info"],
                data.non_tensor_batch["reward_model"],
                scores,
                self._eval_step,
            )
            self._eval_step += 1

        if return_dict:
            return {  # pyright: ignore
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
