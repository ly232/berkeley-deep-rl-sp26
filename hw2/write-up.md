# Write Ups

## Vanilla Policy Gradient

See run results at https://wandb.ai/yang7-cooper-google/cs285_hw2/table, under tag `hw2_part3`.

> Which value estimator has better performance without advantage normalization: the trajectorycentric one, or the one using reward-to-go?

RTG generally outperforms vanilla version.

> Between the two value estimators, why do you think one is generally preferred over the other?

Causality implies past rewards from past actions do not matter for future reward estimates, thus RTG helps to prevent incorrectly awarding or penalizing a current action, when we try to estimate its future reward weight `Q(s, a)`.


> Did advantage normalization help?

Yes. It reduced variance of min/max/avg returns.

> Did the batch size make an impact?

From the experiment, emperically, larger batch size has lower variance for the max return, while it has higher variance for the min and average return.
