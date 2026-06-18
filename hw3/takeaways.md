# DQN and Double-Q Learning

DQN introduces a target network because bootstrapped Q-learning targets are otherwise moving too quickly: the online network would be chasing targets produced by itself.

But vanilla DQN still has overestimation bias because the target uses a max over noisy Q estimates. The action with the largest estimated value is often the one with positive noise, so max_a Q_target(s', a) tends to be too large.

Double DQN addresses this by splitting action selection and action evaluation. The online critic selects the greedy next action, reflecting the current policy, while the target critic evaluates that selected action, preserving the stabilizing role of the target network. This reduces the tendency to directly back up the same noisy value that won the argmax.

## MsPacman DQN log

```
% uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
wandb: WARNING `start_method` is deprecated and will be removed in a future version of wandb. This setting is currently non-functional and safely ignored.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: yang7-cooper (yang7-cooper-google) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/wandb/analytics/sentry.py:268: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/wandb/analytics/sentry.py:268: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/wandb/analytics/sentry.py:279: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/wandb/analytics/sentry.py:268: DeprecationWarning: Read the `app_url` setting from the appropriate Settings object.
  app_url = wandb.util.app_url(tags["base_url"])  # type: ignore[index]
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/wandb/analytics/sentry.py:279: DeprecationWarning: The `Scope.user` setter is deprecated in favor of `Scope.set_user()`.
  self.scope.user = {"email": email}
wandb: Tracking run with wandb version 0.24.2
wandb: Run data is saved locally in /var/folders/mf/3z49lxfd0jgbxbqysr5_g6y40000gn/T/tmpjq_pqtlg/wandb/run-20260613_113552-gk7caugx
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run MsPacman_dqn_sd1_20260613_113551
wandb: ⭐️ View project at https://wandb.ai/yang7-cooper-google/hw3
wandb: 🚀 View run at https://wandb.ai/yang7-cooper-google/hw3/runs/gk7caugx
wandb: Detected [agents] in use.
wandb: Use W&B Weave for improved LLM call tracing. Install Weave with `pip install weave` then add `import weave` to the top of your script.
wandb: For more information, check out the docs at: https://weave-docs.wandb.ai/
Using CPU.
A.L.E: Arcade Learning Environment (version 0.7.5+db37282)
[Powered by Stella]
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/gym/core.py:317: DeprecationWarning: WARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
  deprecation(
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/gym/wrappers/step_api_compatibility.py:39: DeprecationWarning: WARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
  deprecation(
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/gym/utils/passive_env_checker.py:227: DeprecationWarning: WARN: Core environment is written in old step API which returns one bool instead of two. It is recommended to rewrite the environment with new step API. 
  logger.deprecation(
/Users/ly232/github/berkeley-deep-rl-sp26/hw3/.venv/lib/python3.10/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
  if not isinstance(done, (bool, np.bool8)):
 38%|██████████████████████████████████████████████████████▍                                                                                       | 383155/1000000 [34:41:05<13:36:43, 12 38%|███████████████████████████████████████████████████▎                                                                                  | 383157/1000000 [34:41:05<17:03:28, 10.04it/s]100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000000/1000000 [57:56:20<00:00,  4.79it/s]
wandb: 
wandb: 🚀 View run MsPacman_dqn_sd1_20260613_113551 at: https://wandb.ai/yang7-cooper-google/hw3/runs/gk7caugx
wandb: Find logs at: ../../../../../var/folders/mf/3z49lxfd0jgbxbqysr5_g6y40000gn/T/tmpjq_pqtlg/wandb/run-20260613_113552-gk7caugx/logs
```