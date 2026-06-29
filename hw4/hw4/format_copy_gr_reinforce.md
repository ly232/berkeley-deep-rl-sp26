# Format Copy + GR-REINFORCE

```sh
uv run modal run --detach scripts/modal_train.py -- \
    --task format_copy \
    --algo reinforce \
    --output_dir /vol/runs/modal_format_copy_reinforce \
    --steps 51 \
    --batch_size 8 \
    --group_size 6 \
    --min_new_tokens 1 \
    --max_new_tokens 24 \
    --lr 3e-5 \
    --minibatch_size 8 \
    --grad_accum_steps 6 \
    --kl_coef 0.05 \
    --max_grad_norm 0.5 \
    --wandb_enabled \
    --wandb_project llm-rl-hw4 \
    --wandb_name format_copy_reinforce \
    --sample_markdown_log_interval 1 \
    --sample_log_interval 10 \
    --sample_log_n 6 \
    --eval_interval 50 \
    --save_interval 50 \
    --warmup_steps 10
Note that running a local entrypoint in detached mode only keeps the last triggered Modal function alive after the parent process has been killed or disconnected.
✓ Initialized. View run at https://modal.com/apps/yang7-cooper/main/ap-rnJV9khonaMb12KFLr1DAK
✓ Created objects.
├── 🔨 Created mount /Users/ly232/github/berkeley-deep-rl-sp26/hw4/scripts/modal_train.py
├── 🔨 Created mount /Users/ly232/github/berkeley-deep-rl-sp26/hw4
├── 🔨 Created mount /Users/ly232/.netrc
├── 🔨 Created function train_remote.
├── 🔨 Created function eval_remote.
└── 🔨 Created function bundle_submission_remote.
.remote() and .map() calls in detached apps may be canceled when the local caller disconnects. Use .spawn() for detached or background work.wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: yang7-cooper (yang7-cooper-google) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: setting up run ybevtho5
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /vol/wandb/wandb/run-20260629_012900-ybevtho5
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run format_copy_reinforce
wandb: ⭐️ View project at https://wandb.ai/yang7-cooper-google/llm-rl-hw4
wandb: 🚀 View run at https://wandb.ai/yang7-cooper-google/llm-rl-hw4/runs/ybevtho5
[eval][format_copy] phase=baseline_before_first_rl_update step_zero_based=0 starting evaluation over ~64 examples (progress updates every 6 examples).
[eval][format_copy] phase=baseline_before_first_rl_update step_zero_based=0 progress=32/64 (50.0%) elapsed=2.1s eta~2.1s
[eval][format_copy] phase=baseline_before_first_rl_update step_zero_based=0 progress=64/64 (100.0%) elapsed=3.4s eta~0.0s
[eval][format_copy] phase=baseline_before_first_rl_update step_zero_based=0 finished 64 examples in 3.4s (18.94 examples/sec).
train[reinforce|format_copy]:  18%|█▊        | 9/51 [00:23<01:48,  2.58s/it, reward=0.375, kl=0.012, loss=-0.169]⠋ Running (1/1 containers active)... View app at https://modal.com/apps/yang7-cooper/main/ap-rnJV9khonaMb12KFL
train[reinforce|format_copy]:  69%|██████▊   | 35/51 [01:20<00:38,  2.43s/it, reward=1.183, kl=0.232, loss=-0.358]⠋ Running (1/1 containers active)... View app at https://modal.com/apps/yang7-cooper/main/ap-rnJV9khonaMb12KF
                                                                                                                  
                                                                                                                  ess updates every 6 examples).
                                                                                                                  
                                                                                                                  s
train[reinforce|format_copy]: 100%|██████████| 51/51 [02:00<00:00,  2.36s/it, reward=1.271, kl=0.171, loss=-0.110]/sec).
[eval][format_copy] phase=final_after_last_rl_update step_zero_based=51 starting evaluation over ~64 examples (progress updates every 6 examples).
[eval][format_copy] phase=final_after_last_rl_update step_zero_based=51 progress=32/64 (50.0%) elapsed=0.6s eta~0.6s
[eval][format_copy] phase=final_after_last_rl_update step_zero_based=51 progress=64/64 (100.0%) elapsed=1.2s eta~0.0s
[eval][format_copy] phase=final_after_last_rl_update step_zero_based=51 finished 64 examples in 1.2s (54.90 examples/sec).
wandb: updating run metadata
wandb: uploading output.log; uploading wandb-summary.json
wandb: 
wandb: Run history:
wandb:                          eval/examples_per_second_in_last_evaluation_call ▁██
wandb:           eval/format_copy_fraction_completions_containing_answer_xml_tag ▁██
wandb:     eval/format_copy_fraction_completions_that_are_strict_answer_xml_only ▁██
wandb: eval/format_copy_fraction_predicted_number_matches_target_integer_exactly ▁██
wandb:                            eval/format_copy_number_of_evaluation_examples ▁▁▁
wandb:                 eval/number_of_examples_processed_in_last_evaluation_call ▁▁▁
wandb:                             eval/runtime_seconds_for_last_evaluation_call █▁▁
wandb:                            format_copy/completion_contains_answer_xml_tag ▂▁▂▂▂▁▁▃▆▇██████████████████▇████████▇█▇
wandb:     format_copy/completion_is_strictly_only_answer_xml_without_extra_text ▁▁▁▁▁▁▁▂▃▄▇▇████████████▇██▇▇▇█████▇▇█▇█
wandb:                  model/count_total_parameters_including_frozen_base_model ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                                                       +28 ...
wandb: 
wandb: Run summary:
wandb:                          eval/examples_per_second_in_last_evaluation_call 54.90383
wandb:           eval/format_copy_fraction_completions_containing_answer_xml_tag 1
wandb:     eval/format_copy_fraction_completions_that_are_strict_answer_xml_only 1
wandb: eval/format_copy_fraction_predicted_number_matches_target_integer_exactly 1
wandb:                            eval/format_copy_number_of_evaluation_examples 64
wandb:                 eval/number_of_examples_processed_in_last_evaluation_call 64
wandb:                             eval/runtime_seconds_for_last_evaluation_call 1.16567
wandb:                            format_copy/completion_contains_answer_xml_tag 0.97917
wandb:     format_copy/completion_is_strictly_only_answer_xml_without_extra_text 0.95833
wandb:                  model/count_total_parameters_including_frozen_base_model 1562179072.0
wandb:                                                                       +29 ...
wandb: 
wandb: 🚀 View run format_copy_reinforce at: https://wandb.ai/yang7-cooper-google/llm-rl-hw4/runs/ybevtho5
wandb: ⭐️ View project at: https://wandb.ai/yang7-cooper-google/llm-rl-hw4
wandb: Synced 5 W&B file(s), 5 media file(s), 10 artifact file(s) and 0 other file(s)
wandb: Find logs at: /vol/wandb/wandb/run-20260629_012900-ybevtho5/logs
✓ App completed. View run at https://modal.com/apps/yang7-cooper/main/ap-rnJV9khonaMb12KFLr1DAK
(cs285-hw4-llm-rl) ly232@iMacPro hw4 % 
```
