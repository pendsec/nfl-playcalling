"""Policy learning (Step 6): the recommended defensive policy pi*(A|S).

    greedy        — behavior-constrained greedy: argmax_a Q(s,a) restricted to
                    calls with enough behavioral support (pi_b(a|s) >= threshold).
                    The constraint is the simplest form of the pessimism CQL
                    formalizes, and it is the baseline the regression gate scores
                    the candidate policy against.
    conservative  — the shipped policy: the same argmax with a KL-to-behavior
                    penalty scaled by the Q model's own spread, so it departs
                    from the DC only where Q is confidently higher.

CLAUDE.md's Step 6 plans the deeper rungs from here — causal pessimism,
invariant policy learning across teams/seasons, and a hierarchical factored
policy over the larger action space.
"""
