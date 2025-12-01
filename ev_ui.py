import streamlit as st

from ev_model import load_model, estimate_all_values, compute_uma

# --- constants / defaults ---

DEFAULT_WIND = "E"
DEFAULT_ROUND = 1
DEFAULT_HONBA = 0
DEFAULT_RIICHI = 0
DEFAULT_SCORES = [25000, 25000, 25000, 25000]
SEAT_LABELS = ["East", "South", "West", "North"]


# --- init ---


@st.cache_resource
def get_model():
    return load_model()  # uses ev_model.MODEL_PATH


model = get_model()

st.set_page_config(
    page_title="Riichi EV Sandbox",
    layout="centered",
)

st.title("Riichi EV Sandbox")

st.caption(
    "Adjust the round state and scores to see the expected value for each seat "
    "(Tenhou uma 90/45/0/−135). EV is in thousands, relative to 25k + uma."
)

# --- session defaults ---


def ensure_defaults():
    if "wind" not in st.session_state:
        st.session_state["wind"] = DEFAULT_WIND
    if "round" not in st.session_state:
        st.session_state["round"] = DEFAULT_ROUND
    if "honba" not in st.session_state:
        st.session_state["honba"] = DEFAULT_HONBA
    if "riichi" not in st.session_state:
        st.session_state["riichi"] = DEFAULT_RIICHI
    for i in range(4):
        key = f"score_{i}"
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_SCORES[i]


ensure_defaults()

# --- reset button ---

reset_pressed = st.button("Reset to defaults")

if reset_pressed:
    st.session_state["wind"] = DEFAULT_WIND
    st.session_state["round"] = DEFAULT_ROUND
    st.session_state["honba"] = DEFAULT_HONBA
    st.session_state["riichi"] = DEFAULT_RIICHI
    for i in range(4):
        st.session_state[f"score_{i}"] = DEFAULT_SCORES[i]

st.markdown("---")

# --- inputs ---

col_top = st.columns(4)

with col_top[0]:
    wind_label = st.selectbox(
        "Bakaze",
        ["E", "S"],
        key="wind",
    )

with col_top[1]:
    round_num = st.selectbox(
        "Round (1 = East/South 1)",
        [1, 2, 3, 4],
        key="round",
    )

with col_top[2]:
    honba = st.number_input(
        "Honba",
        min_value=0,
        max_value=10,
        step=1,
        key="honba",
    )

with col_top[3]:
    riichi = st.number_input(
        "Riichi sticks",
        min_value=0,
        max_value=10,
        step=1,
        key="riichi",
    )

st.markdown("---")

st.subheader("Scores at start of round")

score_cols = st.columns(4)
scores_pts: list[int] = []

for i in range(4):
    with score_cols[i]:
        s = st.number_input(
            SEAT_LABELS[i],
            min_value=-30000,
            max_value=90000,
            step=1000,
            key=f"score_{i}",
        )
        scores_pts.append(int(s))

st.markdown(
    "<small>Scores are in raw points (e.g. 25000). "
    "EV output is in thousands.</small>",
    unsafe_allow_html=True,
)

# convert to thousands for the model
scores_thousands = [s / 1000.0 for s in scores_pts]

# --- compute model EVs ---

evs_model_th = estimate_all_values(
    model=model,
    wind=wind_label,
    round_num=round_num,
    honba=honba,
    riichi=riichi,
    scores_thousands=scores_thousands,
)

# --- compute baseline EVs (current score + uma if game ended now) ---

uma_pts = compute_uma(scores_pts)                 # in points
uma_th = [u / 1000.0 for u in uma_pts]           # thousands
scores_th = [s / 1000.0 for s in scores_pts]

# baseline EV = (score - 25000)/1000 + uma/1000 = score_th - 25 + uma_th
evs_baseline_th = [
    scores_th[i] - 25.0 + uma_th[i] for i in range(4)
]

# --- display ---

def fmt_ev(x: float) -> str:
    # 1 decimal, with sign
    return f"{x:+.1f}"


st.markdown("---")
st.subheader("Expected value (thousands)")

st.markdown("**Model EV**")
row1 = st.columns(4)
for i in range(4):
    with row1[i]:
        st.metric(
            label=SEAT_LABELS[i],
            value=fmt_ev(evs_model_th[i]),
        )

st.markdown("**Baseline EV if game ended now (current score + uma)**")
row2 = st.columns(4)
for i in range(4):
    with row2[i]:
        st.metric(
            label=SEAT_LABELS[i],
            value=fmt_ev(evs_baseline_th[i]),
            delta=fmt_ev(evs_model_th[i] - evs_baseline_th[i]),
        )

with st.expander("Debug / raw values"):
    st.write("Inputs:")
    st.json(
        {
            "wind": wind_label,
            "round": round_num,
            "honba": honba,
            "riichi": riichi,
            "scores_pts": scores_pts,
        }
    )
    st.write("Model EVs (thousands):", [round(v, 3) for v in evs_model_th])
    st.write("Baseline EVs (thousands):", [round(v, 3) for v in evs_baseline_th])
