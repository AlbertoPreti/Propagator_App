import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import date, datetime, timedelta
import matplotlib.dates as mdates
import plotly.graph_objects as go


def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# =========================================================
# CONSTANTS
# =========================================================
MU_EARTH = 398600.0   # [km^3/s^2]
R_EARTH = 6378.0      # [km]
OMEGA_EARTH = 7.292115900231276e-5  # [rad/s] Earth's rotation rate


# =========================================================
# DEFAULT VALUES - NEOSSATO
# =========================================================
DEFAULTS = {
    # Orbital elements
    "sma_km": 6854.0,
    "ecc": 0.0002334,
    "inc_deg": 28.47,
    "raan_deg": 206.8,
    "aop_deg": 83.44,
    "ta_deg": 276.6,

    # Epoch
    "epoch_date": date.today(),
    "epoch_hour": 9,
    "epoch_minute": 30,
    "epoch_second": 0,

    # Physical / geometrical properties
    "mass_kg": 11258.0,
    "area_m2": 51.5,
    "cd": 2.2,

    # Propagation settings
    "prop_model": "Simple Two Body",
    "time_max_days": 5.0,
}


# =========================================================
# SESSION-STATE HELPERS
# =========================================================
def init_session_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "run_pressed" not in st.session_state:
        st.session_state.run_pressed = False

    if "orbit_result" not in st.session_state:
        st.session_state.orbit_result = None

    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if "error_message" not in st.session_state:
        st.session_state.error_message = None

    if "right_panel_view" not in st.session_state:
        st.session_state.right_panel_view = "orbit"


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state.run_pressed = False
    st.session_state.orbit_result = None
    st.session_state.comparison_result = None
    st.session_state.error_message = None


def collect_user_inputs():
    """Read all current inputs from Streamlit session state."""
    return {
        "sma_km": st.session_state.sma_km,
        "ecc": st.session_state.ecc,
        "inc_deg": st.session_state.inc_deg,
        "raan_deg": st.session_state.raan_deg,
        "aop_deg": st.session_state.aop_deg,
        "ta_deg": st.session_state.ta_deg,
        "epoch_date": st.session_state.epoch_date,
        "epoch_hour": st.session_state.epoch_hour,
        "epoch_minute": st.session_state.epoch_minute,
        "epoch_second": st.session_state.epoch_second,
        "mass_kg": st.session_state.mass_kg,
        "area_m2": st.session_state.area_m2,
        "cd": st.session_state.cd,
        "prop_model": st.session_state.prop_model,
        "time_max_days": st.session_state.time_max_days,
    }


def validate_orbit_geometry(sma_km, ecc):
    """
    Check that the perigee radius is larger than the Earth radius.
    """
    rp_km = sma_km * (1.0 - ecc)
    is_valid = rp_km > R_EARTH + 120
    return is_valid, rp_km


# =========================================================
# COMPUTATION CORE
# =========================================================
def state_vector_from_COE(kepler_elements):
    """
    Converts Classical Orbital Elements (COE) to position and velocity
    state vectors in the ECI frame.

    Parameters
    ----------
    kepler_elements : array_like
        [a, e, i, Omega, omega, nu]
        Units: [km, -, deg, deg, deg, deg]

    Returns
    -------
    r_ECI : ndarray, shape (3, 1)
        Position vector in ECI [km]
    v_ECI : ndarray, shape (3, 1)
        Velocity vector in ECI [km/s]
    """

    a, e, i, Omega, omega, nu = kepler_elements

    i = np.deg2rad(i)
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)
    nu = np.deg2rad(nu)

    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    p = a * (1 - e**2)
    r = p / (1 + e * cos_nu)

    r_pqw = np.array([[r * cos_nu], [r * sin_nu], [0.0]])
    v_pqw = np.array([
        [-np.sqrt(MU_EARTH / p) * sin_nu],
        [ np.sqrt(MU_EARTH / p) * (e + cos_nu)],
        [0.0]
    ])

    L = np.array([
        [
            cos_Omega * cos_omega - sin_Omega * cos_i * sin_omega,
            -cos_Omega * sin_omega - sin_Omega * cos_i * cos_omega,
            sin_Omega * sin_i
        ],
        [
            sin_Omega * cos_omega + cos_Omega * cos_i * sin_omega,
            -sin_Omega * sin_omega + cos_Omega * cos_i * cos_omega,
            -cos_Omega * sin_i
        ],
        [
            sin_i * sin_omega,
            sin_i * cos_omega,
            cos_i
        ]
    ])

    r_ECI = L @ r_pqw
    v_ECI = L @ v_pqw

    return r_ECI, v_ECI


def COE_from_state_vector(r_ECI, v_ECI, tol=1e-10):
    """
    Converts a position and velocity state vector in the ECI frame
    to Classical Orbital Elements (COE).

    Returns
    -------
    a, e, i_deg, RAAN_deg, AOP_deg, TA_deg
    """

    r_ECI = np.asarray(r_ECI).flatten()
    v_ECI = np.asarray(v_ECI).flatten()

    r = np.linalg.norm(r_ECI)
    v = np.linalg.norm(v_ECI)

    # Specific angular momentum
    h_vec = np.cross(r_ECI, v_ECI)
    h = np.linalg.norm(h_vec)
    h_hat = h_vec / h

    # Node vector
    z_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(z_hat, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = (1.0 / MU_EARTH) * (
        (v**2 - MU_EARTH / r) * r_ECI - np.dot(r_ECI, v_ECI) * v_ECI
    )
    e = np.linalg.norm(e_vec)

    # Specific orbital energy
    energy = 0.5 * v**2 - MU_EARTH / r
    if abs(e - 1.0) > 1e-12:
        a = -MU_EARTH / (2.0 * energy)
    else:
        a = np.inf

    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # RAAN
    if n > tol:
        RAAN = np.arctan2(n_vec[1], n_vec[0])
    else:
        RAAN = 0.0

    # Argument of perigee
    if e > tol and n > tol:
        AOP = np.arctan2(
            np.dot(h_hat, np.cross(n_vec, e_vec)),
            np.dot(n_vec, e_vec)
        )
    elif e > tol and n <= tol:
        AOP = np.arctan2(e_vec[1], e_vec[0])
        RAAN = 0.0
    else:
        AOP = 0.0

    # True anomaly
    if e > tol:
        TA = np.arctan2(
            np.dot(h_hat, np.cross(e_vec, r_ECI)),
            np.dot(e_vec, r_ECI)
        )
    else:
        if n > tol:
            TA = np.arctan2(
                np.dot(h_hat, np.cross(n_vec, r_ECI)),
                np.dot(n_vec, r_ECI)
            )
            AOP = 0.0
        else:
            TA = np.arctan2(r_ECI[1], r_ECI[0])
            RAAN = 0.0
            AOP = 0.0

    def to_deg_360(x):
        return np.degrees(x) % 360.0

    i_deg = to_deg_360(i)
    RAAN_deg = to_deg_360(RAAN)
    AOP_deg = to_deg_360(AOP)
    TA_deg = to_deg_360(TA)

    return a, e, i_deg, RAAN_deg, AOP_deg, TA_deg


def TBP(t, Y):
    """
    Two-Body Problem equations of motion in ECI coordinates.

    Parameters
    ----------
    t : float
        Time [s]
    Y : ndarray, shape (6,)
        State vector [rx, ry, rz, vx, vy, vz]

    Returns
    -------
    dYdt : ndarray, shape (6,)
        Time derivative of the state vector
    """

    rx, ry, rz, vx, vy, vz = Y
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    drxdt = vx
    drydt = vy
    drzdt = vz

    dvxdt = -MU_EARTH * rx / r**3
    dvydt = -MU_EARTH * ry / r**3
    dvzdt = -MU_EARTH * rz / r**3

    return np.array([drxdt, drydt, drzdt, dvxdt, dvydt, dvzdt])


def exponential_atmospheric_model(r_ECI):
    """
    Computes the atmospheric density at a given position using an exponential atmospheric model.

    The model uses a piecewise exponential law fitted to tabulated data stored in
    'exponential_atmospheric_data.txt'.
    """

    filepath = get_data_path("exponential_atmospheric_data.txt")
    exponential_atmospheric_data = np.loadtxt(filepath, delimiter=",")

    h_min = exponential_atmospheric_data[:, 0]
    h_max = exponential_atmospheric_data[:, 1]

    # Altitude above spherical Earth
    h_ellp = np.linalg.norm(r_ECI) - R_EARTH

    # Find row corresponding to altitude interval
    row_index = np.where((h_min <= h_ellp) & (h_max > h_ellp))[0]

    if row_index.size > 0:
        row = row_index[0]

        h_0 = exponential_atmospheric_data[row, 2]
        rho_0 = exponential_atmospheric_data[row, 3]
        H = exponential_atmospheric_data[row, 4]

        rho = rho_0 * np.exp(-(h_ellp - h_0) / H)

    elif h_ellp >= h_max[-1]:
        rho = 0.0
    else:
        raise ValueError(f"No atmospheric match found for altitude: {h_ellp:.2f} km")

    return rho


def perturbed_TBP(t, Y, ballistic_coeff, perturbations):
    """
    Computes the derivatives of position and velocity under central gravity
    and selected perturbations:
        - drag
        - J2
    """

    rx, ry, rz, vx, vy, vz = Y
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    acc = np.zeros(3)

    # -------------------------
    # Atmospheric drag
    # -------------------------
    if "drag" in perturbations:
        rho = exponential_atmospheric_model(np.array([rx, ry, rz])) * 1e9

        v_rel_vec = np.array([
            vx + OMEGA_EARTH * ry,
            vy - OMEGA_EARTH * rx,
            vz
        ])
        v_rel = np.linalg.norm(v_rel_vec)

        if v_rel > 0.0:
            a_d = -0.5 * ballistic_coeff * rho * (v_rel**2) * (v_rel_vec / v_rel)
            acc += a_d

    # -------------------------
    # J2 perturbation
    # -------------------------
    if "J2" in perturbations:
        j2 = 1.08262668e-3

        a_j2x = 1.5 * ((MU_EARTH * j2 * R_EARTH**2 * rx) / (r**5)) * ((5 * rz**2 / r**2) - 1)
        a_j2y = 1.5 * ((MU_EARTH * j2 * R_EARTH**2 * ry) / (r**5)) * ((5 * rz**2 / r**2) - 1)
        a_j2z = 1.5 * ((MU_EARTH * j2 * R_EARTH**2 * rz) / (r**5)) * ((5 * rz**2 / r**2) - 3)

        acc += np.array([a_j2x, a_j2y, a_j2z])

    drxdt = vx
    drydt = vy
    drzdt = vz

    dvxdt = -MU_EARTH * rx / r**3 + acc[0]
    dvydt = -MU_EARTH * ry / r**3 + acc[1]
    dvzdt = -MU_EARTH * rz / r**3 + acc[2]

    return np.array([drxdt, drydt, drzdt, dvxdt, dvydt, dvzdt])


def propagate_orbit(user_inputs):
    """
    Build the initial state from the user inputs and propagate the orbit.
    """

    SMA = user_inputs["sma_km"]
    ECC = user_inputs["ecc"]
    INC = user_inputs["inc_deg"]
    RAAN = user_inputs["raan_deg"]
    AOP = user_inputs["aop_deg"]
    TA = user_inputs["ta_deg"]

    prop_model = user_inputs["prop_model"]
    n_day = user_inputs["time_max_days"]

    mass_kg = user_inputs["mass_kg"]
    area_m2 = user_inputs["area_m2"]
    cd = user_inputs["cd"]

    _epoch_date = user_inputs["epoch_date"]
    _epoch_h = user_inputs["epoch_hour"]
    _epoch_m = user_inputs["epoch_minute"]
    _epoch_s = user_inputs["epoch_second"]

    rp_km = SMA * (1.0 - ECC)
    if rp_km <= R_EARTH + 120.0:
        raise ValueError(
            f"Invalid orbit: perigee radius = {rp_km:.2f} km must be greater than "
            f"Earth radius + 120 km = {R_EARTH + 120.0:.2f} km. "
            f"Please modify SMA and/or ECC."
        )

    if prop_model in ["Drag", "Drag + J2"]:
        if mass_kg <= 0.0:
            raise ValueError("Mass must be greater than zero for drag-based propagation.")
        if area_m2 <= 0.0:
            raise ValueError("Cross-sectional area must be greater than zero for drag-based propagation.")
        if cd <= 0.0:
            raise ValueError("Cd must be greater than zero for drag-based propagation.")

    area_km2 = area_m2 * 1e-6
    ballistic_coeff = area_km2 * cd / mass_kg

    r0, v0 = state_vector_from_COE([SMA, ECC, INC, RAAN, AOP, TA])
    Y0 = np.concatenate((r0, v0)).flatten()

    t_max = n_day * 86400.0
    t_span = [0.0, t_max]

    if n_day <= 1.0:
        dt = 1.0
    elif 1.0 < n_day < 7.0:
        dt = 60.0
    else:
        dt = 120.0

    t_eval = np.arange(0.0, t_max + dt, dt)

    if prop_model == "Simple Two Body":
        dyn_fun = lambda t, X: TBP(t, X)

    elif prop_model == "J2":
        dyn_fun = lambda t, X: perturbed_TBP(
            t, X,
            ballistic_coeff=ballistic_coeff,
            perturbations=["J2"]
        )

    elif prop_model == "Drag":
        dyn_fun = lambda t, X: perturbed_TBP(
            t, X,
            ballistic_coeff=ballistic_coeff,
            perturbations=["drag"]
        )

    elif prop_model == "Drag + J2":
        dyn_fun = lambda t, X: perturbed_TBP(
            t, X,
            ballistic_coeff=ballistic_coeff,
            perturbations=["drag", "J2"]
        )

    else:
        raise ValueError(f"Unknown propagation model: {prop_model}")


    sol = solve_ivp(
        fun=dyn_fun,
        t_span=t_span,
        t_eval=t_eval,
        y0=Y0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    r = sol.y[0:3, :].T
    v = sol.y[3:6, :].T

    return {
        "t": sol.t,
        "r": r,
        "v": v,
        "inputs": user_inputs,
        "ballistic_coeff": ballistic_coeff,
    }


# =========================================================
# RESULTS / HISTORY HELPERS
# =========================================================
def build_epoch_datetime(user_inputs):
    """
    Builds the initial epoch as a Python datetime object
    from Streamlit user inputs.
    """
    return datetime(
        year=user_inputs["epoch_date"].year,
        month=user_inputs["epoch_date"].month,
        day=user_inputs["epoch_date"].day,
        hour=int(user_inputs["epoch_hour"]),
        minute=int(user_inputs["epoch_minute"]),
        second=int(user_inputs["epoch_second"]),
    )


def compute_orbital_histories(r, v):
    """
    Computes orbital element histories and derived quantities
    from propagated Cartesian states.
    """
    coe = np.array([COE_from_state_vector(rk, vk) for rk, vk in zip(r, v)])

    a_list = coe[:, 0]
    e_list = coe[:, 1]
    i_list = coe[:, 2]
    RAAN_list = coe[:, 3]
    AOP_list = coe[:, 4]
    TA_list = coe[:, 5]

    rp_list = a_list * (1.0 - e_list)
    ra_list = a_list * (1.0 + e_list)

    return {
        "a_list": a_list,
        "e_list": e_list,
        "i_list": i_list,
        "RAAN_list": RAAN_list,
        "AOP_list": AOP_list,
        "TA_list": TA_list,
        "rp_list": rp_list,
        "ra_list": ra_list,
    }


def wrap_to_180(angle_deg):
    """
    Wrap angle difference to [-180, 180] deg.
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def build_propagation_info(result):
    """
    Build compact propagation info and orbital element summary table.
    """
    r = result["r"]
    v = result["v"]
    t = result["t"]
    inputs = result["inputs"]

    histories = compute_orbital_histories(r, v)
    epoch_dt = build_epoch_datetime(inputs)

    start_dt = epoch_dt
    end_dt = epoch_dt + timedelta(seconds=float(t[-1]))
    duration_days = t[-1] / 86400.0

    a0 = histories["a_list"][0]
    af = histories["a_list"][-1]

    e0 = histories["e_list"][0]
    ef = histories["e_list"][-1]

    i0 = histories["i_list"][0]
    if_ = histories["i_list"][-1]

    RAAN0 = histories["RAAN_list"][0]
    RAANf = histories["RAAN_list"][-1]

    AOP0 = histories["AOP_list"][0]
    AOPf = histories["AOP_list"][-1]

    rp0 = histories["rp_list"][0]
    rpf = histories["rp_list"][-1]

    ra0 = histories["ra_list"][0]
    raf = histories["ra_list"][-1]

    table_rows = [
        {
            "quantity": "SMA [km]",
            "initial": a0,
            "final": af,
            "variation": af - a0,
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
        {
            "quantity": "ECC [-]",
            "initial": e0,
            "final": ef,
            "variation": ef - e0,
            "fmt": "{:+.6e}",
            "fmt_abs": "{:.6e}",
        },
        {
            "quantity": "INC [deg]",
            "initial": i0,
            "final": if_,
            "variation": wrap_to_180(if_ - i0),
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
        {
            "quantity": "RAAN [deg]",
            "initial": RAAN0,
            "final": RAANf,
            "variation": wrap_to_180(RAANf - RAAN0),
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
        {
            "quantity": "AOP [deg]",
            "initial": AOP0,
            "final": AOPf,
            "variation": wrap_to_180(AOPf - AOP0),
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
        {
            "quantity": "rp [km]",
            "initial": rp0,
            "final": rpf,
            "variation": rpf - rp0,
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
        {
            "quantity": "ra [km]",
            "initial": ra0,
            "final": raf,
            "variation": raf - ra0,
            "fmt": "{:+.6f}",
            "fmt_abs": "{:.6f}",
        },
    ]

    return {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "duration_days": duration_days,
        "table_rows": table_rows,
    }

def display_propagation_info(result):
    """
    Display propagation info in a clean layout under the plots,
    using aligned columns instead of a heavy table.
    """
    info = build_propagation_info(result)

    # -------------------------------------------------
    # Top row: time information
    # -------------------------------------------------
    st.markdown("## Propagation Info")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Start date**")
        st.write(info["start_dt"].strftime("%d %b %Y - %H:%M:%S"))

    with col2:
        st.markdown("**Propagation time**")
        st.write(f"{info['duration_days']:.3f} days")

    with col3:
        st.markdown("**End date**")
        st.write(info["end_dt"].strftime("%d %b %Y - %H:%M:%S"))

    st.markdown("")
    st.markdown("## Orbital element summary")

    # -------------------------------------------------
    # Header row
    # -------------------------------------------------
    h1, h2, h3, h4 = st.columns([1.35, 1.15, 1.15, 1.0])

    with h1:
        st.markdown("**Quantity**")
    with h2:
        st.markdown("**Initial Value**")
    with h3:
        st.markdown("**Final Value**")
    with h4:
        st.markdown("**Variation**")

    # -------------------------------------------------
    # Data rows
    # -------------------------------------------------
    for row in info["table_rows"]:
        c1, c2, c3, c4 = st.columns([1.35, 1.15, 1.15, 1.0])

        initial_str = row["fmt_abs"].format(row["initial"])
        final_str = row["fmt_abs"].format(row["final"])
        variation_str = row["fmt"].format(row["variation"])

        with c1:
            st.write(row["quantity"])
        with c2:
            st.write(initial_str)
        with c3:
            st.write(final_str)
        with c4:
            st.write(variation_str)

# =========================================================
# PLOTTING HELPERS
# =========================================================
def build_empty_plot():
    fig = go.Figure()

    n_sphere = 40
    u = np.linspace(0, 2 * np.pi, n_sphere)
    v = np.linspace(0, np.pi, n_sphere)

    x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        showscale=False,
        opacity=0.35,
        colorscale=[[0, "royalblue"], [1, "royalblue"]],
        hoverinfo="skip"
    ))

    fig.add_annotation(
        text="3D orbit plot will appear here after PROPAGATE",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=18)
    )

    lim = 1.8 * R_EARTH

    fig.update_layout(
        title="Satellite Orbit in ECI",
        height=900,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(
                title="X ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            yaxis=dict(
                title="Y ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            zaxis=dict(
                title="Z ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=-1.6, y=-1.4, z=0.9)
            )
        )
    )

    return fig


def build_placeholder_result_plot(title):
    fig = go.Figure()

    fig.add_annotation(
        text="Empty plot",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16)
    )

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


def build_orbit_figure(r):
    fig = go.Figure()

    # =========================================================
    # Propagated orbit
    # =========================================================
    fig.add_trace(go.Scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode="lines",
        name="Propagated Orbit",
        line=dict(width=2.25, color="red")
    ))

    # Initial state
    fig.add_trace(go.Scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode="markers",
        name="Satellite Initial State",
        marker=dict(size=5, color="red")
    ))

    # =========================================================
    # Earth sphere
    # =========================================================
    n_sphere = 40
    u = np.linspace(0, 2 * np.pi, n_sphere)
    v = np.linspace(0, np.pi, n_sphere)

    x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        showscale=False,
        opacity=1.0,
        colorscale=[[0, "royalblue"], [1, "royalblue"]],
        name="Earth",
        hoverinfo="skip"
    ))

    # =========================================================
    # ECI reference frame arrows
    # =========================================================
    axis_len = 1.6 * R_EARTH

    # Arrow shafts
    fig.add_trace(go.Scatter3d(
        x=[0, axis_len], y=[0, 0], z=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_len], z=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_len],
        mode="lines",
        line=dict(width=3, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Arrow heads with cones
    cone_size = 0.04 * axis_len

    fig.add_trace(go.Cone(
        x=[axis_len], y=[0], z=[0],
        u=[1], v=[0], w=[0],
        sizemode="absolute",
        sizeref=cone_size,
        showscale=False,
        colorscale=[[0, "black"], [1, "black"]],
        anchor="tip",
        name="ECI Frame",
        hoverinfo="skip"
    ))

    fig.add_trace(go.Cone(
        x=[0], y=[axis_len], z=[0],
        u=[0], v=[1], w=[0],
        sizemode="absolute",
        sizeref=cone_size,
        showscale=False,
        colorscale=[[0, "black"], [1, "black"]],
        anchor="tip",
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[axis_len],
        u=[0], v=[0], w=[1],
        sizemode="absolute",
        sizeref=cone_size,
        showscale=False,
        colorscale=[[0, "black"], [1, "black"]],
        anchor="tip",
        showlegend=False,
        hoverinfo="skip"
    ))

    # Axis labels I, J, K
    label_pos = 1.05 * axis_len

    fig.add_trace(go.Scatter3d(
        x=[label_pos], y=[0], z=[0],
        mode="text",
        text=["I"],
        textfont=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[label_pos], z=[0],
        mode="text",
        text=["J"],
        textfont=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[label_pos],
        mode="text",
        text=["K"],
        textfont=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # =========================================================
    # Layout
    # =========================================================
    max_orbit = np.max(np.abs(r))
    lim = max(max_orbit, axis_len, R_EARTH) * 1.05

    fig.update_layout(
        title="Satellite Orbit in ECI",
        scene=dict(
            xaxis=dict(
                title="X ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            yaxis=dict(
                title="Y ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            zaxis=dict(
                title="Z ECI [km]",
                range=[-lim, lim],
                showgrid=False,
                zeroline=False,
                showbackground=False,
                showline=False,
                ticks=""
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=1.4, z=0.6)
            )
        ),
        height=900,
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def build_time_history_plot(
    t_dates,
    y_main,
    title,
    ylabel,
    y_ref=None,
    main_label="Selected model",
    ref_label="Simple Two Body",
):
    fig = go.Figure()

    # First: perturbed model
    fig.add_trace(go.Scatter(
        x=t_dates,
        y=y_main,
        mode="lines",
        name=main_label,
        line=dict(color = "orange", width=1.5),
    ))

    # Then: Simple Two Body on top
    if y_ref is not None:
        fig.add_trace(go.Scatter(
            x=t_dates,
            y=y_ref,
            mode="lines",
            name=ref_label,
            line=dict(dash="dash", color = "blue", width=1),
        ))

    fig.update_layout(
        title=title,
        height=375,
        margin=dict(l=60, r=20, t=90, b=50),
        xaxis_title="Time",
        yaxis_title=ylabel,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=0.0
        )
    )

    return fig


def build_rp_ra_plot(
    t_dates,
    rp_main,
    ra_main,
    rp_ref=None,
    ra_ref=None,
    main_label_suffix="Selected model",
    ref_label_suffix="Simple Two Body",
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_dates,
        y=rp_main,
        mode="lines",
        name=rf"rp ({main_label_suffix}) [km]",
        line=dict(color="red", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=t_dates,
        y=ra_main,
        mode="lines",
        name=rf"ra ({main_label_suffix}) [km]",
        line=dict(color="green", width=1.5)
    ))

    if rp_ref is not None and ra_ref is not None:
        fig.add_trace(go.Scatter(
            x=t_dates,
            y=rp_ref,
            mode="lines",
            name=rf"rp ({ref_label_suffix}) [km]",
            line=dict(dash="dash", color = "red", width=1),
        ))

        fig.add_trace(go.Scatter(
            x=t_dates,
            y=ra_ref,
            mode="lines",
            name=rf"ra ({ref_label_suffix}) [km]",
            line=dict(dash="dash", color = "green", width=1),
        ))

    fig.update_layout(
        title="Perigee and Apogee Radius",
        height=375,
        margin=dict(l=60, r=20, t=90, b=50),
        xaxis_title="Time",
        yaxis_title="Radius [km]",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="left",
            x=0.0
        )
    )

    return fig


# =========================================================
# STREAMLIT APP CONFIG
# =========================================================
st.set_page_config(
    page_title="Satellite Propagation",
    layout="wide",
)

init_session_state()

st.title("Satellite Propagation")

left_col, right_col = st.columns([1, 2], gap="large")


# =========================================================
# USER INPUT SECTION
# =========================================================
with left_col:
    st.subheader("User Inputs")

    with st.container(border=True):
        st.markdown("#### Orbital Elements")

        st.number_input(
            "SMA [km]",
            min_value=6578.0,
            max_value=36000.0,
            step=1.0,
            key="sma_km",
            help="Semimajor axis. Allowed range: 6578.0 to 10000.0 km.",
        )

        st.number_input(
            "ECC [-]",
            min_value=0.0,
            max_value=0.9,
            step=0.001,
            format="%.6f",
            key="ecc",
            help="Eccentricity. Allowed range: 0.0 to 0.9.",
        )

        orbit_is_valid, rp_km = validate_orbit_geometry(
            st.session_state.sma_km,
            st.session_state.ecc
        )

        if orbit_is_valid:
            st.caption(
                f"Perigee radius: {rp_km:.2f} km  |  Earth radius: {R_EARTH:.2f} km"
            )
        else:
            st.error(
                f"Invalid orbit: the perigee radius is {rp_km:.2f} km, "
                f"which is not greater than the Earth radius ({R_EARTH:.2f} km). "
                f"Please modify SMA and/or ECC."
            )

        st.number_input(
            "INC [deg]",
            min_value=0.0,
            max_value=180.0,
            step=0.1,
            key="inc_deg",
            help="Inclination. Allowed range: 0.0 to 180.0 deg.",
        )

        st.number_input(
            "RAAN [deg]",
            min_value=0.0,
            max_value=360.0,
            step=0.1,
            key="raan_deg",
            help="Right ascension of the ascending node. Allowed range: 0.0 to 360.0 deg.",
        )

        st.number_input(
            "AOP [deg]",
            min_value=0.0,
            max_value=360.0,
            step=0.1,
            key="aop_deg",
            help="Argument of perigee. Allowed range: 0.0 to 360.0 deg.",
        )

        st.number_input(
            "TA [deg]",
            min_value=0.0,
            max_value=360.0,
            step=0.1,
            key="ta_deg",
            help="True anomaly. Allowed range: 0.0 to 360.0 deg.",
        )

    with st.container(border=True):
        st.markdown("#### Initial Epoch")

        st.date_input(
            "Date",
            key="epoch_date",
        )

        st.markdown("Time [HH:MM:SS]")

        col_h, col_m, col_s = st.columns(3)

        with col_h:
            st.number_input(
                "HH",
                min_value=0,
                max_value=23,
                step=1,
                key="epoch_hour",
            )

        with col_m:
            st.number_input(
                "MM",
                min_value=0,
                max_value=59,
                step=1,
                key="epoch_minute",
            )

        with col_s:
            st.number_input(
                "SS",
                min_value=0,
                max_value=59,
                step=1,
                key="epoch_second",
            )

    with st.container(border=True):
        st.markdown("#### Propagation Model")

        st.selectbox(
            "Select propagation model",
            options=[
                "Simple Two Body",
                "Drag",
                "J2",
                "Drag + J2",
            ],
            key="prop_model",
        )

    if st.session_state.prop_model in ["Drag", "Drag + J2"]:
        with st.container(border=True):
            st.markdown("#### Satellite Physical / Geometrical Properties")

            st.number_input(
                "Mass [kg]",
                min_value=0.1,
                max_value=100000.0,
                #step=0.1,
                value=DEFAULTS.get("mass_kg"),
                key="mass_kg",
            )

            st.number_input(
                "Cross-sectional area [m²]",
                min_value=0.001,
                max_value=10000.0,
                #step=0.001,
                value=DEFAULTS.get("area_m2"),
                key="area_m2",
            )

            st.number_input(
                "Cd [-]",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                value=DEFAULTS.get("cd"),
                key="cd",
                help="Drag coefficient. Suggested value: 2.2",
            )

    with st.container(border=True):
        st.markdown("#### Propagation Time")

        st.number_input(
            "Maximum propagation time [days]",
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            key="time_max_days",
            help="Allowed range: 1.0 to 30.0 days,  1.0 day step",
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        st.button(
            "RESET",
            width='stretch',
            on_click=reset_all,
        )

    with btn_col2:
        propagate_clicked = st.button(
            "PROPAGATE",
            width='stretch',
            disabled=not orbit_is_valid,
        )


# =========================================================
# COMPUTATION TRIGGER SECTION
# =========================================================
if propagate_clicked:
    try:
        user_inputs = collect_user_inputs()
        result = propagate_orbit(user_inputs)

        comparison_result = None
        if user_inputs["prop_model"] != "Simple Two Body":
            comparison_inputs = user_inputs.copy()
            comparison_inputs["prop_model"] = "Simple Two Body"
            comparison_result = propagate_orbit(comparison_inputs)

        st.session_state.orbit_result = result
        st.session_state.comparison_result = comparison_result
        st.session_state.run_pressed = True
        st.session_state.error_message = None
        st.session_state.right_panel_view = "orbit"

    except Exception as exc:
        st.session_state.orbit_result = None
        st.session_state.comparison_result = None
        st.session_state.run_pressed = False
        st.session_state.error_message = str(exc)


# =========================================================
# OUTPUT / PLOT SECTION
# =========================================================
with right_col:
    st.subheader("Output Area")

    view_col1, view_col2 = st.columns(2)

    with view_col1:
        if st.button("3D ORBIT PLOT", width='stretch'):
            st.session_state.right_panel_view = "orbit"

    with view_col2:
        if st.button("RESULTS", width='stretch'):
            st.session_state.right_panel_view = "results"

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if st.session_state.error_message is not None:
        st.warning(st.session_state.error_message)

    if st.session_state.orbit_result is None:
        if st.session_state.right_panel_view == "orbit":
            fig = build_empty_plot()
            st.plotly_chart(fig, width='stretch')
        else:
            row1 = st.columns(2)
            row2 = st.columns(2)

            titles = [
                "Semi-major axis",
                "Eccentricity",
                "Inclination",
                "RAAN",
                "Argument of Perigee",
                "Perigee and Apogee Radius",
            ]

            with row1[0]:
                st.plotly_chart(build_placeholder_result_plot(titles[0]), use_container_width=True)
            with row1[1]:
                st.plotly_chart(build_placeholder_result_plot(titles[1]), use_container_width=True)

            with row2[0]:
                st.plotly_chart(build_placeholder_result_plot(titles[2]), use_container_width=True)
            with row2[1]:
                st.plotly_chart(build_placeholder_result_plot(titles[3]), use_container_width=True)

    else:
        r = st.session_state.orbit_result["r"]
        v = st.session_state.orbit_result["v"]
        t = st.session_state.orbit_result["t"]
        inputs = st.session_state.orbit_result["inputs"]

        if st.session_state.right_panel_view == "orbit":
            fig = build_orbit_figure(r)
            st.plotly_chart(fig, width='stretch')


        elif st.session_state.right_panel_view == "results":
            row1 = st.columns(2)
            row2 = st.columns(2)
            row3 = st.columns(2)

            histories = compute_orbital_histories(r, v)
            epoch_dt = build_epoch_datetime(inputs)
            t_dates = [epoch_dt + timedelta(seconds=float(ti)) for ti in t]

            comparison_result = st.session_state.comparison_result
            comparison_histories = None

            if comparison_result is not None:
                r_ref = comparison_result["r"]
                v_ref = comparison_result["v"]
                comparison_histories = compute_orbital_histories(r_ref, v_ref)

            main_label = inputs["prop_model"]
            ref_label = "Simple Two Body"

            with row1[0]:
                st.plotly_chart(
                    build_time_history_plot(
                        t_dates,
                        histories["a_list"],
                        "Semi-major axis",
                        "SMA [km]",
                        y_ref=comparison_histories["a_list"] if comparison_histories is not None else None,
                        main_label=main_label,
                        ref_label=ref_label,
                    ),
                    use_container_width=True
                )

            with row1[1]:
                st.plotly_chart(
                    build_time_history_plot(
                        t_dates,
                        histories["e_list"],
                        "Eccentricity",
                        "ECC [-]",
                        y_ref=comparison_histories["e_list"] if comparison_histories is not None else None,
                        main_label=main_label,
                        ref_label=ref_label,
                    ),
                    use_container_width=True
                )

            with row2[0]:
                st.plotly_chart(
                    build_time_history_plot(
                        t_dates,
                        histories["i_list"],
                        "Inclination",
                        "INC [deg]",
                        y_ref=comparison_histories["i_list"] if comparison_histories is not None else None,
                        main_label=main_label,
                        ref_label=ref_label,
                    ),
                    use_container_width=True
                )

            with row2[1]:
                st.plotly_chart(
                    build_time_history_plot(
                        t_dates,
                        histories["RAAN_list"],
                        "RAAN",
                        "RAAN [deg]",
                        y_ref=comparison_histories["RAAN_list"] if comparison_histories is not None else None,
                        main_label=main_label,
                        ref_label=ref_label,
                    ),
                    use_container_width=True
                )

            with row3[0]:
                st.plotly_chart(
                    build_time_history_plot(
                        t_dates,
                        histories["AOP_list"],
                        "Argument of Perigee",
                        "AOP [deg]",
                        y_ref=comparison_histories["AOP_list"] if comparison_histories is not None else None,
                        main_label=main_label,
                        ref_label=ref_label,
                    ),
                    use_container_width=True
                )

            with row3[1]:
                st.plotly_chart(
                    build_rp_ra_plot(
                        t_dates,
                        histories["rp_list"],
                        histories["ra_list"],
                        rp_ref=comparison_histories["rp_list"] if comparison_histories is not None else None,
                        ra_ref=comparison_histories["ra_list"] if comparison_histories is not None else None,
                        main_label_suffix=main_label,
                        ref_label_suffix=ref_label,
                    ),
                    use_container_width=True
                )

            display_propagation_info(st.session_state.orbit_result)
