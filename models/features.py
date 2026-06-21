"""Feature encoding shared by XGBoost training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .helper import compute_uma, _wind_to_id
except ImportError:  # Allows direct script execution from models/.
    from helper import compute_uma, _wind_to_id


SUPPORTED_FEATURE_VERSIONS = ("legacy", "v1", "v2")

LEGACY_FEATURE_NAMES = [
    "wind_id",
    "round",
    "honba_bucket",
    "riichi_bucket",
    "seat",
    "rot_score_0_th",
    "rot_score_1_th",
    "rot_score_2_th",
    "rot_score_3_th",
]

V1_EXTRA_FEATURE_NAMES = [
    "self_score_th",
    "self_baseline_ev_th",
    "gap_to_rel1_th",
    "gap_to_rel2_th",
    "gap_to_rel3_th",
    "gap_to_first_th",
    "gap_to_fourth_th",
    "gap_to_next_higher_th",
    "gap_to_next_lower_th",
    "sorted_gap_1_2_th",
    "sorted_gap_2_3_th",
    "sorted_gap_3_4_th",
    "rot_place_0",
    "rot_place_1",
    "rot_place_2",
    "rot_place_3",
    "rot_uma_0_th",
    "rot_uma_1_th",
    "rot_uma_2_th",
    "rot_uma_3_th",
]

V2_EXTRA_FEATURE_NAMES = [
    "self_score_th",
    "score_total_th",
    "gap_to_rel1_th",
    "gap_to_rel2_th",
    "gap_to_rel3_th",
    "self_vs_max_other_th",
    "self_vs_min_other_th",
    "round_index",
    "hands_until_normal_end",
    "dealer_seat",
    "is_dealer",
    "is_east_round",
    "is_south_round",
    "is_all_last",
]

FEATURE_NAMES = {
    "legacy": LEGACY_FEATURE_NAMES,
    "v1": LEGACY_FEATURE_NAMES + V1_EXTRA_FEATURE_NAMES,
    "v2": LEGACY_FEATURE_NAMES + V2_EXTRA_FEATURE_NAMES,
}


def get_feature_names(feature_version: str) -> list[str]:
    try:
        return list(FEATURE_NAMES[feature_version])
    except KeyError as e:
        raise ValueError(
            f"Unsupported feature version: {feature_version!r}. "
            f"Use one of {SUPPORTED_FEATURE_VERSIONS}."
        ) from e


def feature_count(feature_version: str) -> int:
    return len(get_feature_names(feature_version))


def normalize_wind_id(wind: str | int) -> int:
    return _wind_to_id(wind)


def normalize_wind_label(wind: str | int) -> str:
    wind_id = normalize_wind_id(wind)
    return "E" if wind_id == 0 else "S"


def is_supported_wind(wind: str | int) -> bool:
    try:
        normalize_wind_id(wind)
    except ValueError:
        return False
    return True


def compute_places(scores_pts: Sequence[int] | Sequence[float]) -> list[int]:
    if len(scores_pts) != 4:
        raise ValueError("scores must have length 4")

    order = sorted(range(4), key=lambda i: (-scores_pts[i], i))
    places = [0, 0, 0, 0]
    for place, seat in enumerate(order, start=1):
        places[seat] = place
    return places


def baseline_targets_thousands(scores_pts: Sequence[int] | Sequence[float]) -> list[float]:
    uma_pts = compute_uma(scores_pts)
    return [
        (float(scores_pts[i]) + float(uma_pts[i])) / 1000.0
        for i in range(4)
    ]


def final_evs_thousands(
    final_scores_pts: Sequence[int] | Sequence[float],
    final_places: Sequence[int] | None = None,
) -> list[float]:
    if final_places is None:
        final_uma_pts = compute_uma(final_scores_pts)
    else:
        uma_scheme = [90000, 45000, 0, -135000]
        final_uma_pts = [uma_scheme[int(place) - 1] for place in final_places]

    return [
        (float(final_scores_pts[i]) + float(final_uma_pts[i])) / 1000.0 - 25.0
        for i in range(4)
    ]


def encode_state_row(
    *,
    wind: str | int,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands: Sequence[int] | Sequence[float],
    seat: int,
    feature_version: str = "legacy",
) -> np.ndarray:
    if len(scores_thousands) != 4:
        raise ValueError("scores_thousands must have length 4")
    if not 0 <= int(seat) <= 3:
        raise ValueError("seat must be in 0..3")

    wind_id = normalize_wind_id(wind)
    honba_bucket = min(int(honba), 5)
    riichi_bucket = min(int(riichi), 5)

    scores_th = [float(s) for s in scores_thousands]
    scores_pts = [int(round(s * 1000.0)) for s in scores_th]
    seat = int(seat)

    rotated_scores = [scores_th[(seat + k) % 4] for k in range(4)]
    values = [
        float(wind_id),
        float(int(round_num)),
        float(honba_bucket),
        float(riichi_bucket),
        float(seat),
        *rotated_scores,
    ]

    if feature_version == "legacy":
        return np.asarray(values, dtype=np.float32)

    order = sorted(range(4), key=lambda i: (-scores_pts[i], i))
    scores_by_place = [scores_th[i] for i in order]

    if feature_version == "v2":
        max_other = max(scores_th[i] for i in range(4) if i != seat)
        min_other = min(scores_th[i] for i in range(4) if i != seat)
        wind_round_index = wind_id * 4 + int(round_num)
        dealer_seat = (int(round_num) - 1) % 4
        values.extend(
            [
                scores_th[seat],
                sum(scores_th),
                scores_th[seat] - rotated_scores[1],
                scores_th[seat] - rotated_scores[2],
                scores_th[seat] - rotated_scores[3],
                scores_th[seat] - max_other,
                scores_th[seat] - min_other,
                float(wind_round_index),
                float(max(0, 8 - wind_round_index)),
                float(dealer_seat),
                1.0 if seat == dealer_seat else 0.0,
                1.0 if wind_id == 0 else 0.0,
                1.0 if wind_id == 1 else 0.0,
                1.0 if wind_id == 1 and int(round_num) == 4 else 0.0,
            ]
        )
        return np.asarray(values, dtype=np.float32)

    if feature_version != "v1":
        get_feature_names(feature_version)  # Raises the canonical error.

    places = compute_places(scores_pts)
    uma_th = [u / 1000.0 for u in compute_uma(scores_pts)]
    baseline_ev_th = [
        scores_th[i] + uma_th[i] - 25.0
        for i in range(4)
    ]

    self_score = scores_th[seat]
    self_place = places[seat]
    gap_to_next_higher = (
        0.0 if self_place == 1 else self_score - scores_by_place[self_place - 2]
    )
    gap_to_next_lower = (
        0.0 if self_place == 4 else self_score - scores_by_place[self_place]
    )

    rotated_places = [float(places[(seat + k) % 4]) for k in range(4)]
    rotated_uma = [uma_th[(seat + k) % 4] for k in range(4)]

    values.extend(
        [
            self_score,
            baseline_ev_th[seat],
            self_score - rotated_scores[1],
            self_score - rotated_scores[2],
            self_score - rotated_scores[3],
            self_score - scores_by_place[0],
            self_score - scores_by_place[3],
            gap_to_next_higher,
            gap_to_next_lower,
            scores_by_place[0] - scores_by_place[1],
            scores_by_place[1] - scores_by_place[2],
            scores_by_place[2] - scores_by_place[3],
            *rotated_places,
            *rotated_uma,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def encode_state_rows(
    *,
    wind: str | int,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands: Sequence[int] | Sequence[float],
    feature_version: str = "legacy",
) -> np.ndarray:
    return np.stack(
        [
            encode_state_row(
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=scores_thousands,
                seat=seat,
                feature_version=feature_version,
            )
            for seat in range(4)
        ],
        axis=0,
    )
