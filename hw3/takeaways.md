# DQN and Double-Q Learning

DQN introduces a target network because bootstrapped Q-learning targets are otherwise moving too quickly: the online network would be chasing targets produced by itself.

But vanilla DQN still has overestimation bias because the target uses a max over noisy Q estimates. The action with the largest estimated value is often the one with positive noise, so max_a Q_target(s', a) tends to be too large.

Double DQN addresses this by splitting action selection and action evaluation. The online critic selects the greedy next action, reflecting the current policy, while the target critic evaluates that selected action, preserving the stabilizing role of the target network. This reduces the tendency to directly back up the same noisy value that won the argmax.
