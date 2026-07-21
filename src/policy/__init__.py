"""Policy learning (Step 6): the recommended defensive policy pi*(A|S).

    greedy  — behavior-constrained greedy: argmax_a Q(s,a) restricted to calls
              with enough behavioral support (pi_b(a|s) >= threshold). The
              constraint is V1's stand-in for the pessimism CQL formalizes later.

Later versions add CQL/IQL, causal pessimism, invariant policy learning across
teams/seasons, and a hierarchical factored policy over the larger action space.
"""
