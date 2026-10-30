import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ap.params import Patient, Meal, TARGET, HYPO, HYPER, DEFAULT_PATIENT
from ap.model import equilibrium, rk4_step
from ap.controllers import PIDController, ConstrainedPI, LQRObserverController
from ap import tuned


# ---------------------------------------------------------------------------
# Simulation adapters. The underlying Bergman model and controller equations
# remain in ap/model.py and ap/controllers.py.
# ---------------------------------------------------------------------------
@dataclass
class MealEntry:
    name: str
    quantity_g: float
    calories_kcal: float
    carbs_g: float
    sugar_g: float
    protein_g: float = 0.0
    fat_g: float = 0.0
    time_min: float = 0.0


class MealSchedule:
    """Combine user-entered meals into the model's glucose-appearance input."""
    def __init__(self, entries: List[MealEntry]):
        self.entries = entries
        self.t_meal = min((m.time_min for m in entries), default=0.0)

    def D(self, t, patient: Patient):
        return sum(
            Meal(carbs_g=m.carbs_g, t_meal=m.time_min).D(t, patient)
            for m in self.entries
        )


CONTROLLERS = ["Constrained PI (IOB)", "PID", "LQR + observer"]
FOOD_DB_PATH = Path(__file__).with_name("food_database.csv")
FOOD_DB = pd.read_csv(FOOD_DB_PATH).fillna(0)
FOOD_DB["food_label"] = FOOD_DB["food"] + " · " + FOOD_DB["category"]


def nutrition_for(food_name, quantity_g):
    """Scale bundled per-100 g/ml nutrition estimates to the chosen amount."""
    row = FOOD_DB.loc[FOOD_DB["food"] == food_name].iloc[0]
    factor = float(quantity_g) / 100.0
    return {
        "calories_kcal": float(row["kcal_per_100g"]) * factor,
        "carbs_g": float(row["carbs_g_per_100g"]) * factor,
        "sugar_g": float(row["sugar_g_per_100g"]) * factor,
        "protein_g": float(row["protein_g_per_100g"]) * factor,
        "fat_g": float(row["fat_g_per_100g"]) * factor,
        "category": str(row["category"]),
        "data_note": str(row["data_note"]),
    }


def build_controller(kind, patient, ts, target):
    if kind == "PID":
        return PIDController(patient, **tuned.PID, Ts=ts, r=target)
    if kind == "Constrained PI (IOB)":
        return ConstrainedPI(patient, **tuned.PI, Ts=ts, r=target)
    return LQRObserverController(patient, **tuned.LQR, Ts=ts, r=target)


def init_state(patient, controller_kind, target, ts, noise_sd, delay_min, meals, seed=0):
    controller = build_controller(controller_kind, patient, ts, target)
    controller.reset()
    x = equilibrium(patient)
    n_delay = max(0, int(round(delay_min / ts)))
    return {
        "t": 0.0, "x": x, "controller": controller,
        "history_g": [float(x[0])] * (n_delay + 1),
        "meal_schedule": MealSchedule(meals), "ts": ts,
        "dt": min(0.5, ts), "noise_sd": noise_sd, "delay_min": delay_min,
        "rng": np.random.default_rng(seed),
        "t_data": [0.0], "g_data": [float(x[0])],
        "u_data": [float(patient.u_basal)], "i_data": [float(patient.Ib)],
        "iob_data": [0.0], "meal_data": [0.0], "y_data": [float(x[0])],
        "finished": False,
    }


def step_sim(state, patient):
    ts, dt, t, x = state["ts"], state["dt"], state["t"], state["x"]
    n_sub = max(1, int(round(ts / dt)))
    delay_steps = max(0, int(round(state["delay_min"] / ts)))
    state["history_g"].append(float(x[0]))
    idx = max(0, len(state["history_g"]) - 1 - delay_steps)
    y_true = state["history_g"][idx]
    y = y_true + (state["rng"].normal(0.0, state["noise_sd"]) if state["noise_sd"] > 0 else 0.0)
    u = float(state["controller"](t, y))
    meal = state["meal_schedule"]

    for _ in range(n_sub):
        state["t_data"].append(t)
        state["g_data"].append(float(x[0]))
        state["u_data"].append(u)
        state["i_data"].append(float(x[2]))
        state["meal_data"].append(float(meal.D(t, patient)))
        state["y_data"].append(float(y))
        x = rk4_step(t, x, u, dt, patient, meal)
        t += dt

    state["t"], state["x"] = t, x
    if hasattr(state["controller"], "B"):
        iob = max(0.0, float(state["controller"].B))
    else:
        iob = max(0.0, float(patient.VI * (x[2] - patient.Ib)))
    state["iob_data"].append(iob)
    return {
        "glucose": float(x[0]), "insulin": float(u),
        "plasma_insulin": float(x[2]), "iob": iob,
        "measurement": float(y), "time": float(t),
    }


def run_controller(kind, patient, target, ts, noise_sd, delay_min, meals, duration, seed=0):
    state = init_state(patient, kind, target, ts, noise_sd, delay_min, meals, seed=seed)
    max_steps = int(np.ceil(duration / ts)) + 2
    for _ in range(max_steps):
        if state["t"] >= duration:
            break
        step_sim(state, patient)
    return state


def performance_metrics(state, target):
    t = np.asarray(state["t_data"], dtype=float)
    g = np.asarray(state["g_data"], dtype=float)
    u = np.asarray(state["u_data"], dtype=float)
    if len(t) < 2:
        return {"peak": float(g[-1]), "time_in_range": 100.0, "overshoot": 0.0, "settling": None}
    # Data are sampled internally at dt; calculate time in range on those samples.
    tir = float(np.mean((g >= HYPO) & (g <= HYPER)) * 100)
    peak = float(np.max(g))
    overshoot = float(max(0.0, peak - target))
    tol = 10.0
    settling = None
    for i in range(len(g)):
        if np.all(np.abs(g[i:] - target) <= tol):
            settling = float(t[i])
            break
    return {
        "peak": peak, "time_in_range": tir, "overshoot": overshoot,
        "settling": settling, "mean_insulin": float(np.mean(u)),
        "max_insulin": float(np.max(u)),
        "low_samples": int(np.sum(g < HYPO)),
        "high_samples": int(np.sum(g > HYPER)),
    }


def fmt_time(value):
    return "Not settled" if value is None else f"{value:.0f} min"


def glucose_figure(state, target, meals, title="Glucose response"):
    fig = go.Figure()
    fig.add_hrect(y0=HYPO, y1=HYPER, fillcolor="rgba(45, 160, 110, 0.10)", line_width=0, layer="below")
    fig.add_hline(y=target, line_dash="dash", line_color="#F5B942", annotation_text=f"Target {target:.0f}", annotation_position="top left")
    fig.add_hline(y=HYPO, line_dash="dot", line_color="#E76F51", opacity=.8)
    fig.add_hline(y=HYPER, line_dash="dot", line_color="#E76F51", opacity=.8)
    fig.add_trace(go.Scatter(
        x=np.asarray(state["t_data"], dtype=float) / 60.0, y=state["g_data"], mode="lines",
        name="Blood glucose", line=dict(color="#55C2FF", width=3),
        fill="tozeroy", fillcolor="rgba(85,194,255,0.08)",
        hovertemplate="Time: %{x:.2f} hours<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
    ))
    for meal in meals:
        fig.add_vline(x=meal.time_min / 60.0, line_dash="dot", line_color="#C084FC", opacity=.85)
        fig.add_annotation(x=meal.time_min / 60.0, y=1, yref="paper", text=f"🍽 {meal.name}", showarrow=False, yshift=12, font=dict(size=10, color="#D8B4FE"))
    fig.update_layout(
        title=title, height=390, margin=dict(l=8, r=12, t=52, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7F5", family="Inter, sans-serif"),
        xaxis_title="Simulation time (hours)", yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False, range=[0, 6], dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False)
    return fig


def line_figure(x, series, title, x_title, y_title, height=300):
    fig = go.Figure()
    colors = ["#55C2FF", "#A78BFA", "#F5B942", "#5EE0A0"]
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=np.asarray(x, dtype=float) / 60.0, y=values, mode="lines", name=name, line=dict(color=colors[i % len(colors)], width=2.5)))
    fig.update_layout(
        title=title, height=height, margin=dict(l=8, r=12, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7F5", family="Inter, sans-serif"),
        xaxis_title=x_title, yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False, range=[0, 6], dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False)
    return fig



def continuous_glucose_figure(state, target):
    """Glucose trace with a growing elapsed-time axis and live meal markers."""
    fig = go.Figure()
    fig.add_hrect(y0=HYPO, y1=HYPER, fillcolor="rgba(45,160,110,.10)", line_width=0, layer="below")
    fig.add_hline(y=target, line_dash="dash", line_color="#F5B942", annotation_text=f"Target {target:.0f}")
    fig.add_hline(y=HYPO, line_dash="dot", line_color="#E76F51", opacity=.8)
    fig.add_hline(y=HYPER, line_dash="dot", line_color="#E76F51", opacity=.8)
    x_hours = np.asarray(state["t_data"], dtype=float) / 60.0
    fig.add_trace(go.Scatter(
        x=x_hours, y=state["g_data"], mode="lines", name="Simulated glucose",
        line=dict(color="#55C2FF", width=2.8),
        hovertemplate="Elapsed: %{x:.2f} h<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
    ))
    for meal in state["meal_schedule"].entries:
        meal_hour = float(meal.time_min) / 60.0
        fig.add_vline(x=meal_hour, line_dash="dot", line_color="#C084FC", opacity=.8)
        fig.add_annotation(x=meal_hour, y=1, yref="paper", text=f"Meal: {meal.name}",
                           showarrow=False, yshift=10, font=dict(size=9, color="#D8B4FE"))
    now_h = float(state["t"]) / 60.0
    fig.update_layout(
        title="Continuous glucose monitor · simulated", height=390,
        margin=dict(l=8, r=12, t=54, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7F5", family="Inter, sans-serif"),
        xaxis_title="Elapsed simulation time (hours)", yaxis_title="Glucose (mg/dL)",
        hovermode="x unified", showlegend=False,
    )
    # The horizontal axis grows for the full session; it is not reset to 3 or 6 hours.
    fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)",
                     zeroline=False, range=[0, max(0.25, now_h * 1.04)])
    fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False)
    return fig


def continuous_line_figure(state, values_key, title, y_title, color):
    """Continuous single-series chart sharing the growing simulation timeline."""
    fig = go.Figure()
    x_hours = np.asarray(state["t_data"], dtype=float) / 60.0
    values = np.asarray(state[values_key], dtype=float)
    # Some state series are logged once per controller sample; align lengths defensively.
    count = min(len(x_hours), len(values))
    fig.add_trace(go.Scatter(
        x=x_hours[:count], y=values[:count], mode="lines", name=title,
        line=dict(color=color, width=2.6),
        hovertemplate="Elapsed: %{x:.2f} h<br>%{y:.2f}<extra></extra>",
    ))
    now_h = float(state["t"]) / 60.0
    fig.update_layout(
        title=title, height=330, margin=dict(l=8, r=12, t=50, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7F5", family="Inter, sans-serif"),
        xaxis_title="Elapsed simulation time (hours)", yaxis_title=y_title,
        hovermode="x unified", showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)",
                     zeroline=False, range=[0, max(0.25, now_h * 1.04)])
    fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", zeroline=False)
    return fig


@st.fragment(run_every="1s")
def continuous_monitor_fragment(patient, controller_kind, target, noise_sd, delay_min, random_seed):
    """Session-local continuous simulation. One controller sample advances per refresh."""
    st.markdown("<div class='section-kicker'>Independent continuous simulation</div>", unsafe_allow_html=True)
    st.subheader("Continuous glucose & insulin monitor")
    st.caption(
        "This monitor has its own simulation state and does not change the existing 0–6 hour experiment. "
        "It starts from the model's basal equilibrium; it is not connected to a real CGM or insulin pump."
    )

    if "continuous_sim" not in st.session_state:
        st.session_state.continuous_sim = None
    if "continuous_running" not in st.session_state:
        st.session_state.continuous_running = False
    if "continuous_speed" not in st.session_state:
        st.session_state.continuous_speed = 1.0
    if "continuous_config" not in st.session_state:
        st.session_state.continuous_config = None

    control_cols = st.columns([1.2, 1.2, 1.6, 2.0])
    with control_cols[0]:
        start_clicked = st.button("▶ Start / Resume", type="primary", use_container_width=True,
                                  key="continuous_start")
    with control_cols[1]:
        pause_clicked = st.button("Ⅱ Pause", use_container_width=True, key="continuous_pause")
    with control_cols[2]:
        reset_clicked = st.button("↻ Reset monitor", use_container_width=True, key="continuous_reset")
    with control_cols[3]:
        speed_minutes = st.selectbox(
            "Simulation pace", options=[1.0, 2.0, 5.0, 10.0], index=0,
            format_func=lambda value: f"{value:g} simulated min / real second",
            key="continuous_speed_select",
            help="For this monitor, the selected pace also sets its controller sampling interval. Choose before starting; the original experiment settings remain independent.",
            disabled=st.session_state.continuous_sim is not None,
        )

    if reset_clicked:
        st.session_state.continuous_sim = None
        st.session_state.continuous_running = False
        st.session_state.continuous_meal_counter = 0
        st.session_state.continuous_config = None
        st.rerun()

    if start_clicked:
        if st.session_state.continuous_sim is None:
            # Freeze this tab's experiment settings at start so sidebar edits cannot
            # silently change the virtual patient halfway through a run.
            st.session_state.continuous_config = {
                "patient": patient,
                "controller_kind": controller_kind,
                "target": float(target),
                "noise_sd": float(noise_sd),
                "delay_min": float(delay_min),
                "seed": int(random_seed) + 101,
            }
            cfg = st.session_state.continuous_config
            # Keep this state independent from st.session_state.sim used by Live dashboard.
            st.session_state.continuous_sim = init_state(
                cfg["patient"], cfg["controller_kind"], cfg["target"], float(speed_minutes),
                cfg["noise_sd"], cfg["delay_min"], [], cfg["seed"],
            )
            st.session_state.continuous_meal_counter = 0
        st.session_state.continuous_running = True

    if pause_clicked:
        st.session_state.continuous_running = False

    sim = st.session_state.continuous_sim
    if sim is None:
        st.info("Press **Start / Resume** to initialise the virtual patient at its basal equilibrium.")
    else:
        cfg = st.session_state.continuous_config or {
            "patient": patient, "target": float(target), "controller_kind": controller_kind
        }
        monitor_patient = cfg["patient"]
        monitor_target = float(cfg["target"])
        st.caption(
            f"Controller: **{cfg.get('controller_kind', controller_kind)}** · "
            f"Setpoint: **{monitor_target:.0f} mg/dL** · "
            f"Sensor noise: **{cfg.get('noise_sd', noise_sd):g} mg/dL SD** · "
            f"CGM delay: **{cfg.get('delay_min', delay_min):g} min**"
        )
        if st.session_state.continuous_running:
            # Each one-second fragment refresh advances one controller sample.
            # At initialisation, the selected pace also sets this monitor's sample period.
            step_sim(sim, monitor_patient)

        glucose_now = float(sim["x"][0])
        insulin_now = float(sim["u_data"][-1]) if sim["u_data"] else float(patient.u_basal)
        plasma_now = float(sim["x"][2])
        iob_now = float(sim["iob_data"][-1]) if sim["iob_data"] else 0.0
        status = "LOW" if glucose_now < HYPO else "HIGH" if glucose_now > HYPER else "IN RANGE"
        status_color = "#F87171" if status != "IN RANGE" else "#5EE0A0"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulated glucose", f"{glucose_now:.1f} mg/dL", status)
        m2.metric("Insulin delivery command", f"{insulin_now:.2f} mU/min")
        m3.metric("Plasma insulin", f"{plasma_now:.2f} µU/mL")
        m4.metric("IOB estimate", f"{iob_now:.2f} mU")
        st.markdown(
            f"<div class='small-note'>Monitor: <b style='color:{status_color}'>{'RUNNING' if st.session_state.continuous_running else 'PAUSED'}</b> "
            f"· Elapsed simulated time: <b>{sim['t']:.1f} min ({sim['t']/60:.2f} h)</b> "
            f"· Update interval: 1 second · Pace: {float(sim['ts']):g} simulated min/update</div>",
            unsafe_allow_html=True,
        )

        # Food is logged at the current monitor time, not scheduled in advance.
        st.markdown("#### Log a meal now")
        st.caption("Choose the food and portion. The event is inserted at the current simulation time and drives the same meal-appearance model used by the original experiment.")
        with st.form("continuous_meal_form", clear_on_submit=True):
            c_food, c_qty, c_submit = st.columns([2.2, 1, 1])
            live_food = c_food.selectbox("Food", FOOD_DB["food"].tolist(), key="continuous_food")
            live_quantity = c_qty.number_input("Quantity (g/ml)", min_value=1.0, max_value=3000.0,
                                               value=200.0, step=25.0, key="continuous_quantity")
            add_meal_now = c_submit.form_submit_button("＋ Eat now", type="primary", use_container_width=True)
        if add_meal_now:
            nutrition = nutrition_for(live_food, live_quantity)
            meal_entry = MealEntry(
                live_food, float(live_quantity), nutrition["calories_kcal"],
                nutrition["carbs_g"], nutrition["sugar_g"], nutrition["protein_g"],
                nutrition["fat_g"], float(sim["t"]),
            )
            sim["meal_schedule"].entries.append(meal_entry)
            st.session_state.continuous_meal_counter = st.session_state.get("continuous_meal_counter", 0) + 1
            st.success(
                f"Logged {live_food} at {sim['t']:.1f} simulated minutes · "
                f"{nutrition['carbs_g']:.1f} g estimated carbohydrates."
            )

        glucose_col, insulin_col = st.columns(2, gap="medium")
        with glucose_col:
            st.plotly_chart(continuous_glucose_figure(sim, monitor_target),
                            use_container_width=True, key="continuous_glucose_chart")
        with insulin_col:
            st.plotly_chart(continuous_line_figure(
                sim, "u_data", "Insulin delivery commanded by controller", "Infusion (mU/min)", "#A78BFA"),
                use_container_width=True, key="continuous_insulin_chart")
        plasma_col, iob_col = st.columns(2, gap="medium")
        with plasma_col:
            st.plotly_chart(continuous_line_figure(
                sim, "i_data", "Plasma insulin response", "Insulin (µU/mL)", "#5EE0A0"),
                use_container_width=True, key="continuous_plasma_chart")
        with iob_col:
            iob_x = np.linspace(0, float(sim["t"]), len(sim["iob_data"]))
            iob_fig = go.Figure()
            iob_fig.add_trace(go.Scatter(x=iob_x / 60.0, y=sim["iob_data"], mode="lines",
                                         name="IOB", line=dict(color="#F5B942", width=2.5)))
            iob_fig.update_layout(title="Insulin on board estimate", height=330,
                                  margin=dict(l=8,r=12,t=50,b=8),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#DCE7F5"), xaxis_title="Elapsed simulation time (hours)",
                                  yaxis_title="IOB (mU)", hovermode="x unified")
            iob_fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)",
                                 range=[0, max(0.25, float(sim['t']) / 60.0 * 1.04)])
            iob_fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)")
            st.plotly_chart(iob_fig, use_container_width=True, key="continuous_iob_chart")

        if sim["meal_schedule"].entries:
            with st.expander(f"Meal event log · {len(sim['meal_schedule'].entries)} event(s)", expanded=True):
                meal_rows = [{
                    "Elapsed time (min)": round(m.time_min, 1), "Food": m.name,
                    "Quantity (g/ml)": round(m.quantity_g, 1),
                    "Estimated carbs (g)": round(m.carbs_g, 1),
                    "Estimated energy (kcal)": round(m.calories_kcal, 1),
                } for m in sim["meal_schedule"].entries]
                st.dataframe(pd.DataFrame(meal_rows), use_container_width=True, hide_index=True)

        export = pd.DataFrame({
            "elapsed_time_min": sim["t_data"],
            "simulated_glucose_mg_dL": sim["g_data"],
            "insulin_delivery_command_mU_min": sim["u_data"],
            "plasma_insulin_uU_mL": sim["i_data"],
            "cgm_measurement_mg_dL": sim["y_data"],
        })
        st.download_button("↓ Download continuous-monitor CSV", export.to_csv(index=False).encode("utf-8"),
                           file_name="continuous_monitor_simulation.csv", mime="text/csv",
                           key="continuous_export")

        st.warning(
            "Simulation only. The displayed glucose is generated by the Bergman model from an assumed "
            "basal starting state; it is not your measured blood glucose. Insulin delivery is a simulated "
            "controller command only and is not sent to a pump or medical device. Do not use these values "
            "for treatment or dosing decisions."
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Artificial Pancreas | Control Lab", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e8f0fc; --muted:#9aabc3; --line:rgba(155,180,220,.16); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:
 radial-gradient(ellipse at 8% 0%, rgba(31,111,170,.18), transparent 36%),
 radial-gradient(ellipse at 95% 5%, rgba(109,70,190,.13), transparent 30%),
 #08111f; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(8,17,31,.65); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101d30,#0b1422); border-right:1px solid var(--line); }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
h1,h2,h3 { letter-spacing:-.035em; }
.hero { padding:1.45rem 1.6rem; border:1px solid rgba(112,175,230,.22); border-radius:22px;
 background:linear-gradient(120deg,rgba(19,48,76,.88),rgba(20,25,48,.82)); margin-bottom:1.15rem; }
.eyebrow { color:#7dd3fc; text-transform:uppercase; letter-spacing:.17em; font-size:.72rem; font-weight:800; }
.hero-title { font-size:clamp(1.8rem,3vw,2.7rem); line-height:1.12; font-weight:800; margin:.4rem 0 .55rem; }
.hero-sub { color:#a9bbd3; max-width:820px; font-size:.96rem; line-height:1.65; }
.pill { display:inline-block; border:1px solid rgba(94,224,160,.28); background:rgba(94,224,160,.09); color:#8af0b7;
 border-radius:999px; padding:.32rem .65rem; font-size:.75rem; font-weight:700; margin-right:.3rem; }
.metric-card { background:linear-gradient(150deg,rgba(20,37,58,.94),rgba(13,25,41,.94)); border:1px solid var(--line);
 border-radius:17px; padding:1rem 1.1rem; min-height:112px; }
.metric-label { color:#9aabc3; font-size:.77rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#f0f6ff; font-size:1.8rem; line-height:1.4; font-weight:800; font-family:'DM Mono',monospace; }
.metric-help { color:#8ca1bc; font-size:.76rem; }
.section-kicker { color:#7dd3fc; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.16em; margin-bottom:.25rem; }
.soft-panel { border:1px solid var(--line); background:rgba(15,28,46,.65); border-radius:18px; padding:1rem 1.15rem; }
div[data-testid="stForm"] { border:1px solid var(--line); background:rgba(15,28,46,.62); border-radius:18px; padding:1.2rem; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:800; min-height:2.7rem; }
div[data-testid="stMetric"] { background:rgba(18,35,56,.78); border:1px solid var(--line); padding:1rem; border-radius:15px; }
div[data-testid="stMetricLabel"] { color:#9aabc3; }
div[data-testid="stMetricValue"] { font-family:'DM Mono',monospace; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
.small-note { color:#8da2bd; font-size:.82rem; line-height:1.6; }
@media (max-width: 768px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: .7rem; }
  .hero { padding: 1.1rem; border-radius: 16px; }
  .hero-sub { font-size: .9rem; }
  .metric-card { padding: .75rem; }
  div[data-testid="stMetric"] { padding: .65rem; }
  div[data-testid="stMetricValue"] { font-size: 1.25rem; }
  h1 { font-size: 1.65rem !important; }
  h2 { font-size: 1.35rem !important; }
  .stButton button, .stDownloadButton button { width: 100%; }
  [data-testid="stPlotlyChart"] { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">BECE302L · Control Systems Laboratory</div>
  <div class="hero-title">Artificial Pancreas <span style="color:#7dd3fc">/</span> Control Lab</div>
  <div class="hero-sub">Explore meal disturbances and closed-loop glucose regulation using the Bergman minimal model.
  Configure meals, switch controllers, and inspect the simulated response in an interactive workspace.</div>
  <div style="margin-top:1rem"><span class="pill">BERGMAN MODEL</span><span class="pill">CLOSED-LOOP CONTROL</span><span class="pill">SIMULATION ONLY</span></div>
</div>
""", unsafe_allow_html=True)

if "meals" not in st.session_state:
    st.session_state.meals = []
if "sim" not in st.session_state:
    st.session_state.sim = None

with st.sidebar:
    st.markdown("### ◈ Experiment setup")
    st.caption("Set the experiment conditions")
    controller_kind = st.selectbox("Controller", CONTROLLERS, index=0)
    target = st.slider("Glucose setpoint (mg/dL)", 80, 140, int(TARGET), 5)
    duration = 360.0
    st.markdown("<div class='small-note'>Live horizon: <b>6 hours</b> of simulated time over <b>30 seconds</b> in live mode.</div>", unsafe_allow_html=True)
    ts = st.select_slider("CGM sample period", options=[1.0, 2.0, 5.0, 10.0, 15.0], value=5.0, format_func=lambda x: f"{x:g} min")
    with st.expander("Sensor conditions"):
        noise_sd = st.slider("Measurement noise (SD)", 0.0, 10.0, 3.0, 0.5, help="Gaussian sensor noise in mg/dL.")
        delay_min = st.select_slider("CGM delay", options=[0.0, 5.0, 10.0, 15.0, 20.0, 30.0], value=10.0, format_func=lambda x: f"{x:g} min")
        random_seed = st.number_input("Noise seed", min_value=0, max_value=9999, value=7, step=1)
    with st.expander("Patient model parameters"):
        gb = st.number_input("Basal glucose Gb", 60.0, 160.0, float(DEFAULT_PATIENT.Gb), step=5.0)
        ib = st.number_input("Basal insulin Ib", 1.0, 30.0, float(DEFAULT_PATIENT.Ib), step=0.5)
        p1 = st.number_input("p1 · glucose effectiveness", 0.001, 0.2, float(DEFAULT_PATIENT.p1), format="%.5f")
        p2 = st.number_input("p2 · insulin action decay", 0.001, 0.2, float(DEFAULT_PATIENT.p2), format="%.5f")
        p3 = st.number_input("p3 · insulin sensitivity", 1e-7, 1e-3, float(DEFAULT_PATIENT.p3), format="%.7g")
        n = st.number_input("n · insulin clearance", 0.01, 0.5, float(DEFAULT_PATIENT.n), format="%.4f")
    patient = Patient(p1=p1, p2=p2, p3=p3, n=n, Gb=gb, Ib=ib, VI=DEFAULT_PATIENT.VI, VG=DEFAULT_PATIENT.VG)
    st.markdown("---")
    st.markdown(f"<div class='small-note'>Basal infusion<br><b style='color:#e8f0fc;font-size:1.1rem'>{patient.u_basal:.2f} mU/min</b><br><br>Pump constraint: 0–200 mU/min</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note' style='margin-top:1rem'>Changes to experiment settings reset the current run.</div>", unsafe_allow_html=True)

config_signature = (
    controller_kind, float(target), float(duration), float(ts), float(noise_sd), float(delay_min), int(random_seed),
    float(gb), float(ib), float(p1), float(p2), float(p3), float(n)
)
if st.session_state.get("config_signature") != config_signature:
    st.session_state.sim = None
    st.session_state.config_signature = config_signature

tab_dash, tab_meals, tab_compare, tab_continuous, tab_about = st.tabs(["◉  Live dashboard", "＋  Meal planner", "⇄  Controller lab", "∿  Continuous monitor", "ⓘ  Model notes"])

with tab_meals:
    st.markdown("<div class='section-kicker'>Disturbance design</div>", unsafe_allow_html=True)
    st.subheader("Build a meal schedule")
    st.markdown("<div class='small-note'>Choose a food and quantity. Nutrition is estimated automatically, and meals are scheduled one hour apart in the order you add them. Carbohydrates drive the meal appearance model.</div>", unsafe_allow_html=True)
    # Search controls sit outside the form so filtering updates immediately on desktop and mobile.
    filter_cols = st.columns([1.2, 1.5])
    category_options = ["All categories"] + sorted(FOOD_DB["category"].unique().tolist())
    selected_category = filter_cols[0].selectbox("Food category", category_options)
    search = filter_cols[1].text_input("Search foods", placeholder="Rice, dosa, pizza, gulab jamun, cola…")
    matches = FOOD_DB.copy()
    if selected_category != "All categories":
        matches = matches[matches["category"] == selected_category]
    if search.strip():
        matches = matches[matches["food"].str.contains(search.strip(), case=False, na=False)]
    if matches.empty:
        st.warning("No foods match those filters. Clear the search or add a custom food below.")
        matches = FOOD_DB.head(1)
    with st.form("meal_form", clear_on_submit=True):
        food = st.selectbox("Food", matches["food"].tolist(), help="Calories, carbohydrates, estimated sugar, protein, and fat are looked up automatically.")
        quantity = st.number_input("Quantity (g or ml)", min_value=1.0, max_value=3000.0, value=200.0, step=25.0)
        # Assign meal events automatically in one-hour increments: users only enter food and quantity.
        default_meal_time = float(min(len(st.session_state.meals) * 60, 300))
        st.caption(f"Meal timing is assigned automatically: T+{default_meal_time:.0f} minutes for this entry. Add foods in the order you want them scheduled.")
        submitted = st.form_submit_button("＋ Add food to schedule", type="primary", use_container_width=True)
        if submitted:
            nutrition = nutrition_for(food, quantity)
            if nutrition["carbs_g"] <= 0:
                st.warning("This food has no carbohydrate estimate, so it may have little effect in this simplified glucose model.")
            st.session_state.meals.append(MealEntry(food, quantity, nutrition["calories_kcal"], nutrition["carbs_g"], nutrition["sugar_g"], nutrition["protein_g"], nutrition["fat_g"], default_meal_time))
            st.session_state.sim = None
            st.success(f"Added {food} ({quantity:g} g/ml). Nutrition values calculated automatically.")
    with st.expander("Add a custom food to the catalog"):
        st.caption("For mixed dishes, restaurant meals, and local recipes not in the catalog. Values are per 100 g/ml; only the regular meal flow requires food and quantity.")
        with st.form("custom_food_form", clear_on_submit=True):
            custom_name = st.text_input("Food name")
            custom_category = st.selectbox("Category", sorted(FOOD_DB["category"].unique().tolist()) + ["Custom food"])
            cc1, cc2, cc3 = st.columns(3)
            custom_kcal = cc1.number_input("kcal / 100 g", min_value=0.0, value=100.0, step=10.0)
            custom_carbs = cc2.number_input("Carbs g / 100 g", min_value=0.0, value=15.0, step=1.0)
            custom_sugar = cc3.number_input("Sugar g / 100 g", min_value=0.0, value=5.0, step=1.0)
            cc4, cc5 = st.columns(2)
            custom_protein = cc4.number_input("Protein g / 100 g", min_value=0.0, value=3.0, step=1.0)
            custom_fat = cc5.number_input("Fat g / 100 g", min_value=0.0, value=2.0, step=1.0)
            custom_save = st.form_submit_button("Save custom food")
            if custom_save:
                if not custom_name.strip():
                    st.error("Enter a food name.")
                elif custom_sugar > custom_carbs:
                    st.error("Sugar cannot exceed total carbohydrates in this catalog entry.")
                elif custom_name.strip().lower() in FOOD_DB["food"].str.lower().values:
                    st.error("That food name already exists.")
                else:
                    new_row = {"food": custom_name.strip(), "category": custom_category, "kcal_per_100g": custom_kcal, "carbs_g_per_100g": custom_carbs, "sugar_g_per_100g": custom_sugar, "protein_g_per_100g": custom_protein, "fat_g_per_100g": custom_fat, "data_note": "User-entered custom estimate"}
                    FOOD_DB.loc[len(FOOD_DB)] = new_row
                    FOOD_DB.to_csv(FOOD_DB_PATH, index=False)
                    st.success("Custom food saved to food_database.csv. It will be available in future sessions on this deployment if the app filesystem is writable; for permanent cloud storage, commit the updated CSV to GitHub.")
                    st.rerun()
    if st.session_state.meals:
        meals_sorted = sorted(enumerate(st.session_state.meals), key=lambda item: item[1].time_min)
        st.markdown("#### Scheduled meals")
        total_cal = sum(m.calories_kcal for m in st.session_state.meals)
        total_carbs = sum(m.carbs_g for m in st.session_state.meals)
        m1, m2, m3 = st.columns(3)
        m1.metric("Meals scheduled", len(st.session_state.meals))
        m2.metric("Total energy", f"{total_cal:,.0f} kcal")
        m3.metric("Total carbohydrates", f"{total_carbs:.0f} g")
        for original_index, meal in meals_sorted:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(f"**🍽️ {meal.name}**  \n`T+{meal.time_min:.0f} min` · {meal.quantity_g:.0f} g/ml")
                c2.metric("Energy", f"{meal.calories_kcal:.0f} kcal")
                c3.metric("Carbs", f"{meal.carbs_g:.1f} g")
                c4.metric("Sugar", f"{meal.sugar_g:.1f} g")
                st.caption(f"Estimated nutrition · {meal.calories_kcal:.0f} kcal · {meal.protein_g:.1f} g protein · {meal.fat_g:.1f} g fat")
                if st.button("Remove meal", key=f"remove_{original_index}"):
                    st.session_state.meals.pop(original_index)
                    st.session_state.sim = None
                    st.rerun()
        if st.button("Clear entire meal schedule", type="secondary"):
            st.session_state.meals = []
            st.session_state.sim = None
            st.rerun()
    else:
        st.info("Your meal schedule is empty. Add a meal above, then run the experiment from Live dashboard.")

with tab_dash:
    st.markdown("<div class='section-kicker'>Live experiment</div>", unsafe_allow_html=True)
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Glucose regulation")
        st.markdown(f"<div class='small-note'>Controller: <b>{controller_kind}</b> &nbsp;·&nbsp; Setpoint: <b>{target} mg/dL</b> &nbsp;·&nbsp; Duration: <b>6 hours</b></div>", unsafe_allow_html=True)
    with right:
        if st.session_state.meals:
            st.markdown(f"<div class='soft-panel'><div class='metric-label'>Meal schedule</div><div style='font-size:1.2rem;font-weight:800;margin-top:.35rem'>{len(st.session_state.meals)} event(s)</div><div class='metric-help'>{sum(m.carbs_g for m in st.session_state.meals):.0f} g total carbs · {sum(m.calories_kcal for m in st.session_state.meals):.0f} kcal</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='soft-panel'><div class='metric-label'>Meal schedule</div><div style='font-size:1.1rem;font-weight:800;margin-top:.35rem'>No meals added</div><div class='metric-help'>Open Meal planner to add an event.</div></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.2, 1.2, 1])
    run_clicked = b1.button("▶ Run full simulation", type="primary", use_container_width=True, disabled=not st.session_state.meals)
    step_clicked = b2.button("⏭ Step one sample", use_container_width=True, disabled=not st.session_state.meals)
    reset_clicked = b3.button("↻ Reset run", use_container_width=True)
    if reset_clicked:
        st.session_state.sim = None
        st.rerun()
    if step_clicked:
        if st.session_state.sim is None:
            st.session_state.sim = init_state(patient, controller_kind, target, float(ts), float(noise_sd), float(delay_min), st.session_state.meals, int(random_seed))
        if st.session_state.sim["t"] < duration:
            step_sim(st.session_state.sim, patient)
        st.rerun()
    if run_clicked:
        sim = init_state(patient, controller_kind, target, float(ts), float(noise_sd), float(delay_min), st.session_state.meals, int(random_seed))
        progress = st.progress(0, text="Initializing virtual patient…")
        live_metrics = st.empty()
        max_steps = int(np.ceil(duration / ts))
        wall_start = time.monotonic()

        # Fixed chart slots prevent the page from jumping as the live trace updates.
        # The two most important monitors stay side-by-side on desktop and stack on mobile.
        glucose_col, insulin_col = st.columns(2, gap="medium")
        with glucose_col:
            st.markdown("#### Blood glucose")
            live_glucose = st.empty()
        with insulin_col:
            st.markdown("#### Insulin delivery")
            live_insulin = st.empty()
        plasma_col, iob_col = st.columns(2, gap="medium")
        with plasma_col:
            st.markdown("#### Plasma insulin")
            live_plasma = st.empty()
        with iob_col:
            st.markdown("#### Insulin on board")
            live_iob = st.empty()

        for step_index in range(max_steps):
            if sim["t"] >= duration:
                break
            out = step_sim(sim, patient)
            g_now = out["glucose"]
            status = "LOW" if g_now < HYPO else "HIGH" if g_now > HYPER else "IN RANGE"

            with live_metrics.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Glucose", f"{g_now:.1f} mg/dL", status)
                k2.metric("Insulin delivery", f"{out['insulin']:.2f} mU/min")
                k3.metric("Plasma insulin", f"{out['plasma_insulin']:.2f} µU/mL")
                k4.metric("IOB estimate", f"{out['iob']:.1f} mU")

            # Redraw at every controller sample (instead of only 12 times per run).
            # Stable empty placeholders replace the previous chart instead of adding new charts.
            ti = np.linspace(0, sim["t"], len(sim["iob_data"]))
            live_glucose.plotly_chart(
                glucose_figure(sim, target, st.session_state.meals),
                use_container_width=True,
            )
            live_insulin.plotly_chart(
                line_figure(sim["t_data"], {"Insulin delivery (mU/min)": sim["u_data"]},
                            "Insulin delivery", "Simulation time (hours)", "Infusion (mU/min)", 390),
                use_container_width=True,
            )
            live_plasma.plotly_chart(
                line_figure(sim["t_data"], {"Plasma insulin (µU/mL)": sim["i_data"]},
                            "Plasma insulin", "Simulation time (hours)", "Insulin (µU/mL)", 300),
                use_container_width=True,
            )
            live_iob.plotly_chart(
                line_figure(ti, {"Insulin-on-board estimate (mU)": sim["iob_data"]},
                            "Insulin on board estimate", "Simulation time (hours)", "IOB (mU)", 300),
                use_container_width=True,
            )
            progress.progress(
                min(1.0, sim["t"] / duration),
                text=f"Simulated {min(6.0, sim['t'] / 60):.2f} / 6.0 hours · target runtime 30 seconds",
            )

            # Spread the controller samples across 30 seconds. If rendering takes longer,
            # the loop catches up without adding an extra sleep.
            target_elapsed = 30.0 * (step_index + 1) / max_steps
            elapsed = time.monotonic() - wall_start
            if target_elapsed > elapsed:
                time.sleep(target_elapsed - elapsed)

        sim["finished"] = True
        st.session_state.sim = sim
        progress.progress(1.0, text="Simulation complete · 6.00 simulated hours")
        st.success("Experiment finished. The results are saved in this session; reset to run again.")
    if st.session_state.sim is not None:
        sim = st.session_state.sim
        g_now = float(sim["g_data"][-1])
        metrics = performance_metrics(sim, target)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Final glucose", f"{g_now:.1f} mg/dL", "Below 70" if g_now < HYPO else "Above 180" if g_now > HYPER else "In clinical reference band")
        k2.metric("Peak glucose", f"{metrics['peak']:.1f} mg/dL")
        k3.metric("Time 70–180", f"{metrics['time_in_range']:.1f}%")
        k4.metric("Peak excursion", f"{metrics['overshoot']:.1f} mg/dL above setpoint")
        # Keep glucose and insulin delivery visible together after the run as well.
        t = sim["t_data"]
        glucose_col, insulin_col = st.columns(2, gap="medium")
        with glucose_col:
            st.plotly_chart(glucose_figure(sim, target, st.session_state.meals), use_container_width=True, key="glucose_chart")
        with insulin_col:
            st.plotly_chart(line_figure(t, {"Insulin delivery (mU/min)": sim["u_data"]}, "Insulin delivery", "Simulation time (hours)", "Infusion (mU/min)", 390), use_container_width=True, key="insulin_delivery_chart")
        plasma_col, iob_col = st.columns(2, gap="medium")
        with plasma_col:
            st.plotly_chart(line_figure(t, {"Plasma insulin (µU/mL)": sim["i_data"]}, "Plasma insulin", "Simulation time (hours)", "Insulin (µU/mL)", 300), use_container_width=True, key="plasma_insulin_chart")
        ti = np.linspace(0, sim["t"], len(sim["iob_data"]))
        with iob_col:
            st.plotly_chart(line_figure(ti, {"Insulin on board estimate (mU)": sim["iob_data"]}, "Insulin on board estimate", "Simulation time (hours)", "IOB (mU)", 300), use_container_width=True, key="insulin_on_board_chart")
        export = pd.DataFrame({
            "time_min": sim["t_data"],
            "glucose_mg_dL": sim["g_data"],
            "insulin_delivery_mU_min": sim["u_data"],
            "plasma_insulin_uU_mL": sim["i_data"],
            "meal_appearance_mg_dL_min": sim["meal_data"],
            "cgm_measurement_mg_dL": sim["y_data"],
        })
        st.download_button("↓ Download simulation CSV", export.to_csv(index=False).encode("utf-8"), file_name="artificial_pancreas_simulation.csv", mime="text/csv")
        if g_now < HYPO:
            st.error("Simulation alert: glucose ended below 70 mg/dL.")
        elif g_now > HYPER:
            st.warning("Simulation alert: glucose ended above 180 mg/dL.")
        else:
            st.success("Final simulated glucose is within the 70–180 mg/dL reference band.")
    else:
        st.markdown("<div class='soft-panel'><div class='metric-label'>Ready state</div><div style='font-size:1.15rem;font-weight:800;margin:.4rem 0'>Configure → Schedule meals → Run</div><div class='small-note'>Your model starts from its basal equilibrium. Use Step one sample to advance in controlled increments, or run the full simulated horizon and inspect the traces below.</div></div>", unsafe_allow_html=True)

with tab_compare:
    st.markdown("<div class='section-kicker'>Controller benchmark</div>", unsafe_allow_html=True)
    st.subheader("Compare control strategies on the same meal schedule")
    st.markdown("<div class='small-note'>Runs a separate offline simulation for each controller with identical patient parameters, meal schedule, sensor delay, noise level, and random seed. This is a simulated comparison, not a clinical recommendation.</div>", unsafe_allow_html=True)
    compare_button = st.button("⇄ Run controller comparison", type="primary", disabled=not st.session_state.meals, key="compare_btn")
    if compare_button:
        compare_states = {}
        with st.spinner("Running three nonlinear closed-loop simulations…"):
            for kind in CONTROLLERS:
                compare_states[kind] = run_controller(kind, patient, float(target), float(ts), float(noise_sd), float(delay_min), st.session_state.meals, float(duration), int(random_seed))
        st.session_state.compare_states = compare_states
        st.session_state.compare_signature = config_signature + (tuple((m.name, m.carbs_g, m.time_min) for m in st.session_state.meals),)
    if st.session_state.get("compare_states") and st.session_state.get("compare_signature") == config_signature + (tuple((m.name, m.carbs_g, m.time_min) for m in st.session_state.meals),):
        states = st.session_state.compare_states
        fig = go.Figure()
        fig.add_hrect(y0=HYPO, y1=HYPER, fillcolor="rgba(45,160,110,.10)", line_width=0, layer="below")
        fig.add_hline(y=target, line_dash="dash", line_color="#F5B942", annotation_text=f"Target {target}")
        colors = {"Constrained PI (IOB)": "#55C2FF", "PID": "#A78BFA", "LQR + observer": "#5EE0A0"}
        for kind, state in states.items():
            fig.add_trace(go.Scatter(x=np.asarray(state["t_data"], dtype=float) / 60.0, y=state["g_data"], mode="lines", name=kind, line=dict(color=colors[kind], width=2.5)))
        for meal in st.session_state.meals:
            fig.add_vline(x=meal.time_min / 60.0, line_dash="dot", line_color="#C084FC", opacity=.6)
        fig.update_layout(title="Same virtual patient · same meals · controller response", height=430, margin=dict(l=8,r=8,t=50,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#DCE7F5"), xaxis_title="Simulation time (hours)", yaxis_title="Glucose (mg/dL)", hovermode="x unified")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)", range=[0, 6], dtick=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(160,180,210,0.12)")
        st.plotly_chart(fig, use_container_width=True, key="controller_comparison_chart")
        rows = []
        for kind, state in states.items():
            m = performance_metrics(state, target)
            rows.append({
                "Controller": kind,
                "Peak glucose (mg/dL)": round(m["peak"], 1),
                "Time in range (%)": round(m["time_in_range"], 1),
                "Excursion above target (mg/dL)": round(m["overshoot"], 1),
                "Mean insulin (mU/min)": round(m.get("mean_insulin", 0), 2),
                "Max insulin (mU/min)": round(m.get("max_insulin", 0), 2),
                "Low samples (<70)": m.get("low_samples", 0),
                "High samples (>180)": m.get("high_samples", 0),
                "Settling time (±10 mg/dL)": fmt_time(m.get("settling")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Metrics are descriptive summaries of the simulated trajectories. Do not interpret them as evidence of clinical safety or dosing performance.")
    elif not st.session_state.meals:
        st.info("Add one or more meals in Meal planner to enable controller comparison.")

with tab_continuous:
    continuous_monitor_fragment(
        patient, controller_kind, float(target), float(noise_sd), float(delay_min), int(random_seed)
    )

with tab_about:
    st.markdown("<div class='section-kicker'>Model transparency</div>", unsafe_allow_html=True)
    st.subheader("What this simulator represents")
    st.markdown("""
    This app wraps the project's existing **Bergman minimal model** and controller implementations. The original Live dashboard remains a 0–6 simulated-hour experiment. The separate Continuous monitor maintains an expanding elapsed-time trace for the duration of the active browser session.
    The model states represent plasma glucose, remote insulin action, and plasma insulin. Meal events
    create a glucose-appearance disturbance, while the selected controller calculates the total simulated
    infusion rate from the CGM measurement.
    """)
    st.latex(r"\dot G = -p_1(G-G_b)-XG+D(t)")
    st.latex(r"\dot X = -p_2X+p_3(I-I_b)")
    st.latex(r"\dot I = -n(I-I_b)+\frac{u-u_{\mathrm{basal}}}{V_I}")
    st.markdown("#### Food database & nutrition estimates")
    st.markdown(f"The bundled searchable catalog contains **{len(FOOD_DB)} foods** across Indian meals, continental/bakery, fast food/snacks, sweets/desserts, drinks, and fruit/basic foods. Nutrition is stored per 100 g/ml and scaled automatically to the quantity you enter. The catalog is a practical reference/estimate set informed by public food-composition references; mixed dishes, restaurant portions, brands, and recipes vary substantially.")
    st.markdown("Sources for reference: [USDA FoodData Central](https://fdc.nal.usda.gov/) and [ICMR–National Institute of Nutrition, Indian Food Composition Tables 2017](https://www.nin.res.in/ebooks/IFCT2017.pdf). This bundled catalog is not a verbatim bulk copy of either database; entries for mixed dishes and packaged foods are estimates.")
    st.markdown("#### How inputs are used")
    st.markdown("""
    - **Carbohydrates (g):** drive the model's meal glucose-appearance disturbance.
    - **Calories, estimated sugar, protein, fat and portion size:** calculated from the food catalog and displayed as nutrition estimates; carbohydrates drive the meal disturbance in this simplified model.
    - **CGM period, delay and noise:** affect the controller's sampled measurement.
    - **Controller:** PID, constrained PI with IOB projection, or LQR with observer.
    """)
    st.warning("**Coursework simulation only — not a medical device.** The parameters are illustrative literature-style values, not patient data. Do not use this app to make treatment decisions or calculate real insulin doses.")

st.markdown("---")
st.markdown("<div class='small-note'>ARTIFICIAL PANCREAS CONTROL LAB · BECE302L · Python / SciPy / Streamlit · Simulation only</div>", unsafe_allow_html=True)
