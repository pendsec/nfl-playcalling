"""Policy learning (Step 6): the recommended defensive policy pi*(A|S).

    greedy        — behavior-constrained greedy: argmax_a Q(s,a) restricted to
                    calls with enough behavioral support (pi_b(a|s) >= threshold).
                    The constraint is the simplest form of the pessimism CQL
                    formalizes, and it is the baseline the regression gate scores
                    the candidate policy against.
    conservative  — the same argmax with a KL-to-behavior penalty scaled by the
                    Q model's own spread, so it departs from the DC only where Q
                    is confidently higher.
    safe          — keeps the coordinator's own mix unless a supported call's
                    advantage over it clears a bootstrap lower bound (and, for
                    the box, an allowance for the coordinator's read,
                    `read_bias`). `learn_policy` dispatches on `policy.type`.
    hierarchical  — the factored policy: an alignment, then a coverage given
                    it, scored together per charted dropback.

The policies' decision rules (`act`) live with their dataclasses in
`src.schemas.policy`; the modules here build them from fitted models.
"""
