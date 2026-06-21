# HW3 Retrospective

This homework built up from DQN to SAC, and the main thread was learning how value targets, policy updates, and stability tricks fit together in deep RL.

## Big Takeaways

### DQN separates value estimation from action selection

The DQN critic is not itself a policy. It maps an observation to Q-values for all discrete actions:

```text
s -> [Q(s, a_0), Q(s, a_1), ...]
```

The policy is the rule layered on top:

```text
greedy:         argmax_a Q(s, a)
epsilon-greedy: usually argmax, sometimes random
```

This clarified why `DQNAgent` can inherit `nn.Module` without defining `forward()`: it is more of an agent/controller that owns networks, optimizers, target updates, and action-selection methods. The actual forward pass lives in the critic network.

### Batch dimensions are everywhere

The line:

```python
observation = ptu.from_numpy(np.asarray(observation))[None]
```

adds a batch dimension. A single CartPole observation with shape `(4,)` becomes `(1, 4)`, and an Atari stacked observation `(4, 84, 84)` becomes `(1, 4, 84, 84)`.

This was an early reminder that PyTorch networks usually expect batched inputs, even for one observation.

### Bellman targets require careful indexing

For DQN, the critic returns:

```text
qa_values.shape = (batch_size, num_actions)
```

But the replay buffer stores one action per transition:

```text
action.shape = (batch_size,)
```

So training needs `gather`, not `max`, to select `Q(s, action_taken)`. The target side also needs `gather` once the next action is chosen.

Example:

```python
qa_values = torch.tensor([
    [1.0, 5.0, 2.0],
    [7.0, 3.0, 4.0],
])

actions = torch.tensor([1, 2])

selected = torch.gather(
    input=qa_values,
    dim=1,
    index=actions.unsqueeze(1),
)

# selected is:
# tensor([[5.0],
#         [4.0]])

q_values = selected.squeeze(1)
# tensor([5.0, 4.0])
```

The `actions` tensor starts as shape `(batch_size,)`, but `gather` expects `index` to have the same number of dimensions as `input`, so `actions.unsqueeze(1)` turns it into `(batch_size, 1)`.

Important distinction:

```text
prediction side: Q_online(s, action_taken)
target side:     reward + gamma * Q_target(s', next_action)
```

### Double DQN splits selection and evaluation

Vanilla DQN uses the target critic for both:

```text
argmax_a Q_target(s', a)
Q_target(s', selected_action)
```

Double DQN instead uses:

```text
selected_action = argmax_a Q_online(s', a)
target_value    = Q_target(s', selected_action)
```

The key reason is overestimation bias. A max over noisy estimates tends to select actions with positive noise. Double DQN reduces that bias by avoiding the use of the exact same noisy estimate for both winning the argmax and setting the backed-up value.

One subtle confusion was about why the online network selects and the target network evaluates, rather than the reverse. The principle is:

```text
online critic: current greedy policy / action selection
target critic: stable bootstrapped value
```

### SAC extends the same critic backbone to continuous actions

SAC still uses replay, bootstrapping, target critics, and MSE critic loss. The main difference is that continuous actions make this impossible:

```text
max_a Q(s', a)
```

So SAC samples the next action from the actor:

```text
a' ~ pi(. | s')
target uses Q_target(s', a')
```

This made SAC feel like DQN's off-policy Bellman machinery plus a learnable continuous-action policy.

### Entropy changes the objective

SAC optimizes a soft objective:

```text
maximize Q(s, a) + alpha * H(pi(. | s))
```

For critic targets, the entropy bonus appears in the next-state value:

```text
Q_target(s', a') + alpha * H(pi(. | s'))
```

For actor loss, because optimizers minimize:

```text
loss = -Q(s, a) - alpha * entropy
```

The homework's entropy implementation used a Monte Carlo estimate:

```text
H(pi(. | s)) = E[-log pi(a | s)]
```

with one sampled action per batch element. A confusion here was whether to return a scalar mean or a vector. The right helper returns shape `(batch_size,)`; callers decide when to average.

### `sample()` vs `rsample()` matters

This was one of the biggest SAC lessons.

`sample()` gives a random sample but does not preserve a gradient path through the sampled value. `rsample()` uses the reparameterization trick, conceptually:

```text
epsilon ~ N(0, I)
action = mu + sigma * epsilon
```

or with tanh squashing:

```text
action = tanh(mu + sigma * epsilon)
```

In code, there is no need to manually model `epsilon`; use:

```python
action = action_distribution.rsample()
```

The important distinction:

```text
critic target under no_grad: sample vs rsample mostly does not matter
actor loss:                  rsample is needed
```

A concrete bug showed up when entropy used `sample()`: entropy collapsed toward a negative value instead of approaching the expected sanity-check region. Switching entropy sampling to `rsample()` fixed it because the entropy bonus was reused in the actor loss and needed to affect actor parameters.

### Continuous entropy is not discrete entropy

There was a useful clarification around entropy scales.

For a discrete uniform distribution over `N` outcomes:

```text
H = log(N)
```

For a continuous uniform distribution over interval length `L`:

```text
H = log(L)
```

So a uniform distribution over `[-1, 1]` has:

```text
H = log(2) ~= 0.69
```

This helped explain the sanity-check entropy expectation.

### Target entropy scales with action dimension, not action count

SAC often initializes:

```python
target_entropy = -action_dim
```

The negative sign comes from log-probability conventions. `action_dim` is not the number of possible actions; it is the dimensionality of a continuous action vector.

For example:

```text
HalfCheetah action_dim = 6
```

means an action is a 6D continuous vector, not one of six choices. Entropy roughly adds across independent dimensions, so the heuristic scales linearly with action dimension, not as `log(action_dim)`.

### Alpha tuning is a scalar optimization problem

The learned temperature is represented with:

```python
log_alpha
alpha = exp(log_alpha)
```

Optimizing `log_alpha` keeps alpha positive.

The alpha loss uses the current policy's log probability as a fixed measurement:

```python
alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
```

The `.detach()` is important because the alpha update should adjust alpha only, not the actor.

Another small implementation detail: `log_alpha` should be a float32 `nn.Parameter`, not float16. Tensor Core speedups matter for large matrix multiplications, not a single scalar parameter with Adam optimizer state.

### Multiple critics are conservative redundancy

SAC supports multiple critics so Section 3.6 can use clipped double-Q:

```text
min(Q1(s, a), Q2(s, a))
```

The critics see the same replay data and are trained with the same objective, but because they are separate function approximators with different initializations and optimizer trajectories, their approximation errors differ.

Taking the minimum is not "averaging out" noise. It intentionally creates a conservative estimate so the actor is less able to exploit one critic's accidental overestimate.

## Confusions Resolved

- Whether `make_critic` returns a policy. It does not; it returns a Q-network. The policy is the argmax or epsilon-greedy rule.
- Why `DQNAgent` does not define `forward()`. The agent is a container/controller; the critic defines the network forward pass.
- Why `[None]` is used on observations. It adds a batch dimension.
- Why DQN training uses `gather`. We need the Q-value for the action actually taken, not the max action.
- Why vanilla DQN overestimates. The max over noisy estimates tends to choose positive noise.
- Why Double DQN uses online for action selection and target for value evaluation. Online is current; target is stable.
- Why SAC has multiple critics. They enable conservative clipped double-Q backups.
- Why entropy cannot simply call `.entropy()` for tanh-transformed Gaussians. PyTorch may not provide analytic entropy for the transformed distribution.
- Why entropy helper should return `(batch_size,)`, not a scalar.
- Why `rsample()` is necessary for actor learning and entropy gradients.
- Why target entropy is `-action_dim`, not `-log(action_dim)`.
- Why `get_temperature()` should return a tensor when alpha is learned, and only convert to `.item()` for logging.
- Why alpha loss detaches `log_prob`. Alpha tuning should not update actor parameters.
