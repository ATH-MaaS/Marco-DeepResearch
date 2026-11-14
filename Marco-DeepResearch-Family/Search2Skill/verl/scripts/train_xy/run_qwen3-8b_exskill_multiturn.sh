#!/bin/bash
# Search2Skill GRPO training entry

set -x

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export HYDRA_FULL_ERROR=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/scripts/train_xy/trainer_config"

for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
        export "$arg"
    fi
done

export MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set to the base model directory}"
export DATA_PATH="${DATA_PATH:?DATA_PATH must point to the directory containing train.parquet and test.parquet}"
export SAVE_PATH="${SAVE_PATH:?SAVE PATH must set}"
export PROJECT_NAME="${PROJECT_NAME:-search2skill_grpo}"
export JOB_NAME="${JOB_NAME:-search2skill_training}"
export EXSKILL_CONFIG_PATH="${EXSKILL_CONFIG_PATH:-$PROJECT_DIR/../config/config.yaml}"
export PYTHONPATH="$PROJECT_DIR/..:$PYTHONPATH"
export EXSKILL_CACHE_DIR="${EXSKILL_CACHE_DIR:-$PROJECT_DIR/cache/exskill}"
export EXSKILL_RAW_PAGE_CACHE_ENABLED="${EXSKILL_RAW_PAGE_CACHE_ENABLED:-true}"
export EXSKILL_RAW_PAGE_CACHE_DIR="${EXSKILL_RAW_PAGE_CACHE_DIR:-$EXSKILL_CACHE_DIR/raw_page_cache_v2}"
export EXSKILL_GOOGLE_SEARCH_CACHE_DIR="${EXSKILL_GOOGLE_SEARCH_CACHE_DIR:-$EXSKILL_CACHE_DIR/google_search_cache_v2}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
mkdir -p "$EXSKILL_CACHE_DIR"

# Auto-discover all *test.parquet files under DATA_PATH for validation
VAL_FILES_LIST=""
for f in "${DATA_PATH}"/*test.parquet; do
    [ -f "$f" ] || continue
    if [ -z "$VAL_FILES_LIST" ]; then
        VAL_FILES_LIST="$f"
    else
        VAL_FILES_LIST="${VAL_FILES_LIST},$f"
    fi
done
if [ -z "$VAL_FILES_LIST" ]; then
    echo "ERROR: No *test.parquet files found in ${DATA_PATH}" >&2
    exit 1
fi
echo "Discovered val files: ${VAL_FILES_LIST}"

#    actor_rollot_ref.actor.use_dynamic_bsz=True \
 #  actor_rollout_ref.rollout.trace.backend=weave \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='exskill_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-64} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-2048} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-8192} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    data.train_files=${DATA_PATH}/train.parquet \
    "data.val_files=[${VAL_FILES_LIST}]" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE:-1e-6} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-False} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}  \
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP} \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.65} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.over_sample_rate=0.0 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.exskill_config_path="${EXSKILL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=exskill_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path="$PROJECT_DIR/scripts/train_xy/reward/exskill_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPU_PER_NODE} \
    trainer.nnodes=${WORLD_SIZE:-1} \
    trainer.save_freq=${SAVE_FREQ:-${STEPS_PER_EPOCH:-1000000}} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-1} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-20} \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.debug_rollout_dump_dir="${DEBUG_ROLLOUT_DUMP_DIR:-$SAVE_PATH/debug_rollouts}" \
    trainer.debug_rollout_dump_freq=${DEBUG_ROLLOUT_DUMP_FREQ:-1} \
    trainer.debug_rollout_dump_samples=${DEBUG_ROLLOUT_DUMP_SAMPLES:-50} \
    trainer.successful_skill_rollout_dump_path="${SUCCESSFUL_SKILL_ROLLOUT_DUMP_PATH:-$SAVE_PATH/successful_skill_rollouts.jsonl}" \
    ${RESUME_FROM_PATH:+trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_FROM_PATH}"}
