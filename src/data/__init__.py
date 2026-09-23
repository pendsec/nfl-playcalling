"""Data layer (Step 1): play-by-play ingestion and (S, A, R) feature engineering.

    load      — pulls nflfastR play-by-play (cached parquet) OR simulates plays
                from a *known* SCM used to validate the causal machinery.
    features  — turns raw pbp into the pre-snap State / Action / Reward table
                every downstream model consumes, enforcing the no-post-snap-leak
                and time-aware-split discipline.
"""
