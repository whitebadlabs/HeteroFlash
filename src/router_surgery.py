#!/usr/bin/env python3
"""Pure-Python experimental GLM-5.3-Flash sigmoid/top-k router algebra.

Reference semantics: HF `Glm5NextTextTopkRouter.forward`, Nov 2026:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py

This implements one token's *routing decisions and weights*, NOT experts, training,
or a model-quality result. Includes optional experimental k_ref denominator.
GLM-5.3-Flash official checkpoint has n_group=topk_group=1, k=8, scale=2.5.
Tie behavior can differ from CUDA torch.topk, so avoid exact-tie assertions.
"""
import math


def route(logits: list[float], correction_bias: list[float], *, k_active=8,
          k_reference=None, scale=2.5, num_groups=1, topk_groups=1):
    if len(logits) != len(correction_bias) or not logits:
        raise ValueError('logits and correction_bias must have identical nonempty length')
    n=len(logits)
    k_reference=k_reference or k_active
    if not (1 <= k_active <= k_reference <= n):
        raise ValueError('require 1 <= k_active <= k_reference <= experts')
    if n % num_groups or not (1 <= topk_groups <= num_groups):
        raise ValueError('invalid group partition')
    score=[1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x)) for x in logits]
    select=[score[i]+correction_bias[i] for i in range(n)]
    width=n//num_groups
    group_scores=[]
    for g in range(num_groups):
        sg=sorted(select[g*width:(g+1)*width],reverse=True)
        group_scores.append(sum(sg[:2]))
    allowed_groups=set(sorted(range(num_groups),key=lambda g:(-group_scores[g],g))[:topk_groups])
    candidates=[i for i in range(n) if i//width in allowed_groups]
    if len(candidates)<k_reference:
        raise ValueError('reference k exceeds allowed group capacity')
    ordered=sorted(candidates,key=lambda i:(-select[i],i))
    active=ordered[:k_active]
    denominator=sum(score[i] for i in ordered[:k_reference]) + 1e-20
    weights=[scale * score[i]/denominator for i in active]
    return {'expert_ids':active, 'weights':weights, 'reference_ids':ordered[:k_reference],
            'sum_weights':sum(weights)}
