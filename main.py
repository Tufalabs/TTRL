"""
Main entry point for Test-Time Reinforcement Learning (TTRL).
Follows the example project pattern with dataclasses for configuration.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import Any
from typing import Dict

import hydra
import numpy as np
import polars as pl
import ray
from omegaconf import OmegaConf
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.ray_trainer import Role
from verl.utils import hf_processor
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.workers.reward_manager.self_reward import decode_batch
from verl.workers.reward_manager.self_reward import encode_chat
from verl.workers.reward_manager.self_reward import to_chat_format

from data.format_for_ttrl import SYSTEM_PROMPT
from data.format_for_ttrl import TASK_PROMPT_DEFINITE
from data.format_for_ttrl import TASK_PROMPT_INDEFINITE
from data.format_for_ttrl import FlatTree
from data.format_for_ttrl import Variant
from data.format_for_ttrl import flatten_trees_from_dir
from data.format_for_ttrl import is_definite_integral_valid
from data.format_for_ttrl import is_indefinite_integral_valid
from data.format_for_ttrl import is_valid_sympy
from data.format_for_ttrl import parse_variant
from data.format_for_ttrl import to_verl_format
from legacy.integration_numeric import FormalStatus
from reward.self_reward import ResultsLogger
from reward.self_reward import SelfRewardManager
from reward.utils import compute_score_definite
from reward.utils import compute_score_indefinite
from reward.utils import extract_candidate_solution
from utils.misc import make_dataclass_from_dict


@dataclass(frozen=True)
class TTRLSpec:
    trees_dir: str
    data_output_dir: str
    num_eval: int
    pass_at_k_params: dict


@dataclass(frozen=True)
class Config:
    ttrl: TTRLSpec


def is_valid_variant(variant: Variant, max_timeout: float = 0.1):
    valid_syntax, integrand_sp = is_valid_sympy(variant)

    if not valid_syntax:
        return False

    if variant.is_definite:
        valid_integral = is_definite_integral_valid(
            integrand_sp, variant.var, variant.lb, variant.ub, max_timeout=max_timeout
        )
    else:
        valid_integral = is_indefinite_integral_valid(
            integrand_sp, variant.var, max_timeout=max_timeout
        )

    return valid_integral


def make_split(flat_tree: FlatTree, num_eval: int):
    datums = []
    for index, variant in enumerate(flat_tree.variants):
        datum = to_verl_format(variant, index, flat_tree.tree_id)
        datums.append(datum)

    eval_datums = []
    for index in range(num_eval):
        base_variant = parse_variant(flat_tree.base_question)
        if base_variant is None:
            raise ValueError("Could not parse eval integral.")

        if not is_valid_variant(base_variant):
            raise ValueError(
                f"Formal verifier cannot evaluate {flat_tree.base_question}"
            )

        datum = to_verl_format(base_variant, index, flat_tree.tree_id)
        eval_datums.append(datum)

    train_df = pl.from_dicts(datums)
    eval_df = pl.from_dicts(eval_datums)

    return train_df, eval_df


def pass_at_k(n, c, k):
    """
    OpenAi Code for pass@k.
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate(
    base_question: str,
    actor_wg: RayWorkerGroup,
    tokenizer: PreTrainedTokenizerBase,
    k: int,
    sampling_params: Dict[str, Any],
):
    assert sampling_params["n"] >= k
    variant_parsed = parse_variant(base_question)

    if variant_parsed is None:
        raise ValueError(f"Could not parse the integral {base_question} for pass@k.")

    if not is_valid_variant(variant_parsed):
        raise ValueError(f"Formal verifier cannot find a solution for {base_question}")

    if variant_parsed.is_definite:
        prompt = TASK_PROMPT_DEFINITE.format(
            integrand=variant_parsed.integrand,
            var=variant_parsed.var,
            lb=variant_parsed.lb,
            ub=variant_parsed.ub,
        )
    else:
        prompt = TASK_PROMPT_INDEFINITE.format(
            integrand=variant_parsed.integrand, var=variant_parsed.var
        )

    chat = to_chat_format(SYSTEM_PROMPT, prompt)
    query = encode_chat(chat, tokenizer, 10000, "error")

    # HACK: We want to reuse the initialized worker group for
    # sglang. But its generate sequence method must get a batch
    # size compatible with its world_size.
    request = DataProto.from_single_dict(
        collate_fn([query for _ in range(actor_wg.world_size)])
    )
    request.meta_info.update({"sampling_params": sampling_params})
    response_bath = actor_wg.generate_sequences(request)
    responses = decode_batch(response_bath, tokenizer)
    responses = [extract_candidate_solution(response) for response in responses]
    N = len(responses)

    correct = 0
    for response in responses:
        if variant_parsed.is_definite:
            _, status = compute_score_definite(
                candidate=response,
                ground_truth=variant_parsed.integrand,
                lb=variant_parsed.lb,
                ub=variant_parsed.ub,
                gt_var=variant_parsed.var,
            )
        else:
            _, status = compute_score_indefinite(
                candidate=response,
                ground_truth=variant_parsed.integrand,
                gt_var=variant_parsed.var,
            )

        if status == FormalStatus.OK:
            correct += 1

    return pass_at_k(N, correct, k)


@hydra.main(config_path="configs", version_base=None)
def main(config):
    # Get project specific args
    custom_config = OmegaConf.masked_copy(config, ["ttrl"])
    custom_config = make_dataclass_from_dict(Config, custom_config)
    ttrl_conf = custom_config.ttrl

    # Process each question directory
    flat_trees = flatten_trees_from_dir(ttrl_conf.trees_dir, max_time=0.001)

    for flat_tree in flat_trees:
        if not ray.is_initialized():
            print("Running for flat_tree", flat_tree.tree_id)

            ray_context = ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "RAY_DEBUG": "legacy",
                    }
                },
                num_cpus=config.ray_init.num_cpus,
            )

            # Create parquet files
            output_dir = os.path.join(ttrl_conf.data_output_dir, flat_tree.tree_id)
            os.makedirs(output_dir, exist_ok=True)

            train_df, eval_df = make_split(flat_tree, ttrl_conf.num_eval)
            train_parquet = os.path.join(output_dir, "train.parquet")
            eval_parquet = os.path.join(output_dir, "eval.parquet")
            train_df.write_parquet(train_parquet)
            eval_df.write_parquet(eval_parquet)

            runner = TaskRunner.remote()
            ray.get(
                runner.run.remote(
                    config, ttrl_conf, train_parquet, eval_parquet, flat_tree
                )
            )

            ray_context.disconnect()

            time.sleep(2)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(
        self,
        config: OmegaConf,
        ttrl_config: TTRLSpec,
        train_parquet: str,
        eval_parquet: str,
        flat_tree: FlatTree,
    ):
        """Run training for a single question and return checkpoint path."""
        exp_name = config["trainer"]["experiment_name"]
        config["trainer"]["experiment_name"] = f"{exp_name}_{flat_tree.tree_id}"

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Instantiate tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        # Define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            from verl.workers.fsdp_workers import CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker
            from verl.workers.megatron_workers import CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Use reference model
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # Set up reward function and loggers
        judge_sampling_params = {"n": 1}
        reward_fn = SelfRewardManager(
            tokenizer=tokenizer, judge_sampling_params=judge_sampling_params
        )

        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_path = os.path.join(
            config.tufa_custom.val_logger.logs_dir,
            f"{config.trainer.experiment_name}_{current_date}",
            f"{flat_tree.tree_id}",
        )
        os.makedirs(exp_path)

        val_logger = ResultsLogger(exp_path, "eval")
        val_reward_fn = SelfRewardManager(
            tokenizer=tokenizer,
            val_logger=val_logger,
            judge_sampling_params=judge_sampling_params,
        )

        # Create datasets
        train_dataset = create_rl_dataset(
            train_parquet, config.data, tokenizer, processor
        )
        val_dataset = create_rl_dataset(eval_parquet, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Create trainer
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        trainer.init_workers()

        print("Computing metric pass@k")
        base_metrics: Dict[int, float] = {}
        for k in [1, 5, 10, 20]:
            base_metrics[k] = evaluate(
                flat_tree.base_question,
                trainer.actor_rollout_wg,
                tokenizer,
                k,
                ttrl_config.pass_at_k_params,
            )

        pprint(base_metrics)
        with open(os.path.join(exp_path, "base_metrics.json"), "w") as f:
            json.dump(base_metrics, f)

        trainer.fit()

        print("Computing final metric pass@k")
        final_metrics: Dict[int, float] = {}
        for k in [1, 5, 10, 20]:
            final_metrics[k] = evaluate(
                flat_tree.base_question,
                trainer.actor_rollout_wg,
                tokenizer,
                k,
                ttrl_config.pass_at_k_params,
            )

        pprint(final_metrics)
        with open(os.path.join(exp_path, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f)


if __name__ == "__main__":
    main()
