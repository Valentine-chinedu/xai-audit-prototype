def explanation_confidence_score(stability: float, fidelity: float, latency_ms: float,
                                 latency_target_ms: float = 200.0) -> float:
    """
    Returns a 0..1 score.
    - stability: 0..1
    - fidelity: usually can be <0..1+, we clamp to 0..1
    - latency penalty: if slower than target, score reduced.
    """
    fid = max(0.0, min(1.0, fidelity))
    stab = max(0.0, min(1.0, stability))

    # latency factor: 1 if within target, decays afterwards
    if latency_ms <= latency_target_ms:
        lat_factor = 1.0
    else:
        lat_factor = max(0.0, latency_target_ms / latency_ms)

    # Weighted blend (tweakable, but defendable)
    score = 0.45 * stab + 0.45 * fid + 0.10 * lat_factor
    return max(0.0, min(1.0, score))
