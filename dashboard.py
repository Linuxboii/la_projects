"""
dashboard.py — Streamlit Dashboard for Chaos Theory Visualizations
===================================================================

Interactive web application for exploring the dynamics of two canonical
chaotic systems:

1. **Logistic Map** — a discrete-time population model that exhibits the
   period-doubling route to chaos.
2. **Lorenz System** — a continuous-time ODE system that produces the
   iconic butterfly-shaped strange attractor.

Controls
--------
Use the sidebar to navigate between visualisations and to tune every
numerical parameter exposed by the underlying modules.  Each page
provides sliders, number inputs, and toggle switches that let you
explore how changing parameters affects the system's behaviour.

Dependencies
------------
- streamlit
- numpy, scipy, matplotlib (via the project modules)
- common_styles (local module for theming)

Usage
-----
Run from the project root::

    streamlit run dashboard.py

Then open the URL shown in the terminal (usually http://localhost:8501).

Module structure
----------------
::

    LA_project/
    ├── dashboard.py          <-- this file
    ├── common_styles.py      <-- shared theming and helper functions
    ├── logistic/
    │   ├── __init__.py
    │   ├── logistic_map.py   <-- core map and orbit generation
    │   ├── bifurcation.py    <-- bifurcation diagram
    │   ├── lyapunov.py       <-- Lyapunov exponent spectrum
    │   └── time_series.py    <-- time-series panels
    ├── lorenz/
    │   ├── __init__.py
    │   ├── lorenz_solver.py  <-- numerical integrator
    │   ├── lorenz_3d_plot.py <-- 3D attractor rendering
    │   └── sensitivity_plot.py  <-- Butterfly Effect plot
    └── requirements.txt

Author
------
Generated for the LA_project.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from lorenz.lorenz_3d_plot import plot_lorenz
from lorenz.sensitivity_plot import plot_sensitivity
from logistic.bifurcation import plot_bifurcation
from logistic.lyapunov import plot_lyapunov
from logistic.time_series import plot_time_series, plot_single_time_series


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="Chaos Theory Visualization Lab",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦋 Chaos Theory Visualization Lab")
st.markdown(
    """
    Explore how **simple deterministic equations** produce incredibly
    complex, unpredictable behaviour.  Use the sidebar controls to tune
    every numerical parameter and watch the system respond in real time.
    """
)


# ============================================================================
# Sidebar — navigation and global controls
# ============================================================================

page = st.sidebar.selectbox(
    "Choose a visualisation",
    [
        "Lorenz Attractor",
        "Butterfly Effect",
        "Logistic Time Series",
        "Bifurcation Diagram",
        "Lyapunov Exponent",
        "LA — System Solver",
        "LA — Iterative Convergence",
    ],
    help="Select which chaos phenomenon to explore.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Parameter controls")


# ============================================================================
# Page 1 — Lorenz Attractor
# ============================================================================

if page == "Lorenz Attractor":
    st.header("Lorenz Attractor")
    st.markdown(
        r"""
        The Lorenz system was originally developed as a simplified weather
        model (Lorenz, 1963).  Despite being fully deterministic, the
        trajectories never repeat and remain confined to a fractal set
        — the **strange attractor**.

        $$
        \dot{x} = \sigma (y - x),\qquad
        \dot{y} = x (\rho - z) - y,\qquad
        \dot{z} = x y - \beta z
        $$
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        sigma = st.slider(
            r"$\sigma$ (Prandtl number)", 1.0, 20.0, 10.0, 0.5,
            help="Viscosity / thermal diffusivity ratio.  Classical value = 10.",
        )
        rho = st.slider(
            r"$\rho$ (Rayleigh number)", 1.0, 50.0, 28.0, 0.5,
            help="Temperature difference driving convection.  Classical value = 28.",
        )
        beta = st.slider(
            r"$\beta$ (aspect ratio)", 0.1, 4.0, 8.0 / 3.0, 0.01,
            format="%.3f",
            help="Physical geometry parameter.  Classical value = 8/3 ≈ 2.667.",
        )

    with col2:
        t_end = st.slider(
            "Integration time  $t_{\\max}$", 5, 100, 40, 5,
            help="Longer times reveal more of the attractor structure.",
        )
        num_points = st.select_slider(
            "Number of points", options=[500, 1000, 2000, 5000, 10000, 20000],
            value=10000,
            help="More points = smoother trajectory, slower render.",
        )
        elev = st.slider("Camera elevation (°)", 0, 90, 25, 1)
        azim = st.slider("Camera azimuth (°)", 0, 360, 45, 1)

    colormap = st.selectbox(
        "Colour map",
        ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
        index=0,
    )

    col1m, col2m = st.columns([1, 3])
    with col1m:
        show_fp = st.checkbox("Show fixed points", value=True)
        show_eq = st.checkbox("Show equations", value=True)
        show_cbar = st.checkbox("Show colour bar", value=True)
    with col2m:
        x0_num = st.number_input(
            r"$x_0$", value=0.0, format="%.2f",
            help="Initial x-coordinate.",
        )
        y0_num = st.number_input(
            r"$y_0$", value=1.0, format="%.2f",
            help="Initial y-coordinate.",
        )
        z0_num = st.number_input(
            r"$z_0$", value=1.05, format="%.2f",
            help="Initial z-coordinate.",
        )

    if st.button("🔄 Generate Lorenz Attractor", type="primary"):
        with st.spinner("Solving Lorenz system and rendering 3D plot…"):
            fig = plot_lorenz(
                initial_conditions=(x0_num, y0_num, z0_num),
                t_end=t_end,
                num_points=num_points,
                sigma=sigma, rho=rho, beta=beta,
                elevation=elev, azimuth=azim,
                colormap=colormap,
                show_fixed_points=show_fp,
                show_equation=show_eq,
                show_colorbar=show_cbar,
            )
            st.pyplot(fig)

    st.markdown("---")
    st.markdown(
        r"""
        **Key insight**: The trajectory never intersects itself — it
        winds around the two fixed points (the "butterfly wings") in an
        aperiodic sequence.  The colour gradient from blue to red shows
        the flow of time.
        """
    )


# ============================================================================
# Page 2 — Butterfly Effect
# ============================================================================

elif page == "Butterfly Effect":
    st.header("Butterfly Effect")

    st.markdown(
        r"""
        Two nearly identical initial conditions diverge **exponentially**.
        Here the reference trajectory starts at :math:`(0, 1, 1.05)` and
        a second trajectory is perturbed by a tiny amount in one
        coordinate.  The plot shows their Euclidean distance over time
        on a logarithmic scale.

        $$
        \delta(t) \sim \delta_0 \, e^{\lambda t}
        $$
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        pert = st.slider(
            "Perturbation magnitude  $\\delta_0$",
            min_value=1e-8, max_value=1e-3, value=1e-5,
            format="%.0e",
            help="Smaller perturbations take longer to diverge.",
        )
        pert_axis = st.selectbox(
            "Perturbed coordinate",
            options=["x", "y", "z"],
            index=2,
            help="Which component of the initial condition to perturb.",
        )
        axis_map = {"x": 0, "y": 1, "z": 2}
        sigma = st.slider(r"$\sigma$", 1.0, 20.0, 10.0, 0.5)

    with col2:
        t_end = st.slider(
            "Integration time  $t_{\\max}$", 5, 100, 40, 5,
        )
        num_points = st.select_slider(
            "Number of points", options=[2000, 5000, 10000, 20000],
            value=10000,
        )
        rho = st.slider(r"$\rho$", 1.0, 50.0, 28.0, 0.5)
        beta = st.slider(
            r"$\beta$", 0.1, 4.0, 8.0 / 3.0, 0.01, format="%.3f",
        )

    with st.expander("Advanced options"):
        show_fit = st.checkbox("Show exponential fit", value=True)
        show_sat = st.checkbox("Show saturation line", value=True)
        show_eq = st.checkbox("Show equation", value=True)

    if st.button("🔄 Generate Butterfly Effect", type="primary"):
        with st.spinner("Integrating two nearby trajectories…"):
            fig = plot_sensitivity(
                perturbation=pert,
                axis=axis_map[pert_axis],
                t_end=t_end,
                num_points=num_points,
                sigma=sigma, rho=rho, beta=beta,
                show_exponential_fit=show_fit,
                show_saturation_line=show_sat,
                show_equation=show_eq,
            )
            st.pyplot(fig)

    st.markdown("---")
    st.markdown(
        r"""
        **Key insight**: The orange dashed line shows an exponential fit
        :math:`e^{\lambda t}`.  Once the distance saturates (horizontal
        dotted line) the trajectories are as far apart as two arbitrary
        points on the attractor — predictability is completely lost.
        """
    )


# ============================================================================
# Page 3 — Logistic Time Series
# ============================================================================

elif page == "Logistic Time Series":
    st.header("Logistic Map Time Series")
    st.markdown(
        r"""
        $$
        x_{n+1} = r \, x_n \, (1 - x_n)
        $$

        Observe how the asymptotic dynamics change as the growth parameter
        *r* increases from a stable fixed point to fully developed chaos.
        """
    )

    mode = st.radio(
        "Display mode",
        ["4-panel overview (default r-values)", "Single custom r-value"],
        horizontal=True,
        help="Overview shows four canonical regimes side by side.",
    )

    if mode == "4-panel overview (default r-values)":
        iterations = st.slider("Iterations per panel", 20, 300, 100, 10)

        if st.button("🔄 Generate Time Series", type="primary"):
            with st.spinner("Generating logistic-map orbits…"):
                fig = plot_time_series(
                    iterations=iterations,
                    show_scatter=True,
                    show_regime_label=True,
                )
                st.pyplot(fig)

        st.markdown("---")
        st.markdown(
            """
            | r | Regime | Behaviour |
            |---|--------|-----------|
            | 2.8 | Fixed point | Rapid convergence to x* ≈ 0.643 |
            | 3.2 | Period-2 | Alternates between two values |
            | 3.5 | Period-4 | Period-doubling to 4-cycle |
            | 3.9 | Chaos | Aperiodic, never repeats |
            """
        )

    else:  # Single custom r-value
        col1, col2 = st.columns(2)
        with col1:
            r = st.slider(r"$r$", 2.0, 4.0, 3.82843, 0.0001, format="%.4f",
                          help="Growth parameter.  r ≈ 3.83 gives a period-3 window.")
            x0 = st.number_input(r"$x_0$", 0.01, 0.99, 0.5, 0.01,
                                 help="Initial population.")
        with col2:
            iterations = st.slider("Iterations to display", 50, 500, 200, 10)
            transient = st.slider("Transient (burn-in)", 0, 5000, 1000, 100,
                                  help="Steps discarded before the observable orbit.")

        if st.button("🔄 Generate Single Time Series", type="primary"):
            with st.spinner("Computing orbit…"):
                fig = plot_single_time_series(
                    r=r, x0=x0,
                    iterations=iterations,
                    transient=transient,
                )
                st.pyplot(fig)

        st.markdown("---")
        st.markdown(
            rf"""
            **Try these r-values:**
            - **r = 3.0** — first period-doubling bifurcation
            - **r = 3.44949** — second period-doubling
            - **r = 3.56995** — chaos onset (Feigenbaum point)
            - **r = 3.82843** — period-3 window
            - **r = 3.90** — fully chaotic
            """
        )


# ============================================================================
# Page 4 — Bifurcation Diagram
# ============================================================================

elif page == "Bifurcation Diagram":
    st.header("Bifurcation Diagram")
    st.markdown(
        r"""
        The iconic branching diagram of the logistic map.  At each *r*
        value, the system is iterated until transients die out, then
        the asymptotic orbit values are plotted.

        $$
        x_{n+1} = r \; x_n \; (1 - x_n)
        $$
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        r_min = st.slider(r"$r_{\min}$", 2.0, 3.8, 2.4, 0.05,
                          help="Left edge of the parameter range.")
        r_max = st.slider(r"$r_{\max}$", 2.5, 4.0, 4.0, 0.05,
                          help="Right edge of the parameter range.")

    with col2:
        num_r = st.select_slider(
            "Number of *r* values",
            options=[1000, 2000, 5000, 8000],
            value=5000,
            help="More r-values = sharper diagram, but slower to generate.",
        )
        transient = st.select_slider(
            "Transient steps",
            options=[500, 1000, 2000, 5000],
            value=2000,
        )

    with col3:
        keep = st.select_slider(
            "Points per *r*",
            options=[100, 200, 500, 1000],
            value=500,
            help="More points per r-value reveals finer attractor structure.",
        )
        density = st.checkbox("Density colouring (slower)", value=True,
                              help="Uses a 2D histogram to colour by local density.")
        transitions = st.checkbox("Show transition lines", value=True)

    if r_min >= r_max:
        st.error("r_min must be < r_max.  Adjust the sliders.")
    else:
        if st.button("🔄 Generate Bifurcation Diagram", type="primary"):
            with st.spinner(
                f"Generating {num_r:,} r-values × {keep} points "
                f"(transient = {transient})…"
            ):
                fig = plot_bifurcation(
                    r_min=r_min, r_max=r_max,
                    num_r=num_r,
                    transient=transient,
                    keep=keep,
                    color_by_density=density,
                    show_transitions=transitions,
                )
                st.pyplot(fig)

    st.markdown("---")
    st.markdown(
        r"""
        **Key transitions**: the vertical dashed lines mark
        *r* = 3 (first period-doubling), *r* ≈ 3.45 (period-4),
        *r* ≈ 3.57 (chaos onset) and *r* ≈ 3.83 (period-3 window).

        **Try zooming in**: set *r_min* = 3.5, *r_max* = 3.6 and observe
        the self-similar structure — the diagram is a fractal.
        """
    )


# ============================================================================
# Page 5 — Lyapunov Exponent
# ============================================================================

elif page == "Lyapunov Exponent":
    st.header("Lyapunov Exponent")
    st.markdown(
        r"""
        The Lyapunov exponent :math:`\lambda(r)` quantitatively separates
        stable (periodic) from chaotic dynamics.

        $$
        \lambda = \frac{1}{N} \sum_{n=1}^{N} \ln\bigl| r(1 - 2x_n) \bigr|
        $$

        - **:red[Red shading]** — :math:`\lambda > 0` → **chaotic**
        - **:green[Green shading]** — :math:`\lambda < 0` → **stable**
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        r_min = st.slider(r"$r_{\min}$", 2.0, 3.8, 2.4, 0.05)
        r_max = st.slider(r"$r_{\max}$", 2.5, 4.0, 4.0, 0.05)
        num_r = st.select_slider(
            "Number of *r* samples",
            options=[1000, 2000, 3000, 5000],
            value=3000,
        )

    with col2:
        transient = st.select_slider(
            "Transient steps", options=[500, 1000, 2000], value=1000,
        )
        iterations = st.select_slider(
            "Iterations per estimate", options=[500, 1000, 2000, 5000],
            value=2000,
            help="More iterations = better λ estimate, slower computation.",
        )
        transitions = st.checkbox("Show transition lines", value=True)

    if r_min >= r_max:
        st.error("r_min must be < r_max.  Adjust the sliders.")
    else:
        if st.button("🔄 Generate Lyapunov Spectrum", type="primary"):
            with st.spinner(
                f"Computing {num_r:,} Lyapunov exponents "
                f"({iterations} iterations each)…"
            ):
                fig = plot_lyapunov(
                    r_min=r_min, r_max=r_max,
                    num_r=num_r,
                    transient=transient,
                    iterations=iterations,
                    show_transitions=transitions,
                )
                st.pyplot(fig)

    st.markdown("---")
    st.markdown(
        r"""
        **Key insight**: Wherever λ(r) crosses from negative to positive,
        the logistic map transitions from periodic to chaotic behaviour.
        The chaotic fraction quantifies how much of the parameter range
        is "unpredictable".
        """
    )


# ============================================================================
# Page — Linear Algebra: System Solver
# ============================================================================

elif page == "LA — System Solver":
    import numpy as np
    import pandas as pd

    from matrix.presets import list_presets, get_preset, PRESETS
    from matrix.linear_system import (
        classify_system, solve_inverse, solve_cramer, solve_gaussian,
        SingularMatrixError,
    )
    from matrix.plot_planes import plane_figure
    from matrix.dashboard_helpers import system_from_editor

    st.header("Linear Algebra — System Solver")
    st.markdown(
        r"""
        Each equation $a x + b y + c z = d$ is a **plane**. The solution of the
        system is where the three planes meet. Edit the coefficients or pick a
        preset, and compare three exact solving methods.
        """
    )

    preset = st.sidebar.selectbox("Preset system", list_presets())
    st.sidebar.caption(PRESETS[preset]["note"])
    A0, b0 = get_preset(preset)

    df = pd.DataFrame(
        np.column_stack([A0, b0]), columns=["x", "y", "z", "= b"]
    )
    edited = st.data_editor(df, key=f"sys_{preset}", use_container_width=True)
    A, b = system_from_editor(edited)

    info = classify_system(A, b)
    st.subheader(f"Solution type: **{info['type'].upper()}**")
    st.write(
        f"rank(A) = {info['rank_A']}, rank([A|b]) = {info['rank_Ab']}, "
        f"det(A) = {info['det']:.3f}"
    )

    solution = None
    if info["type"] == "unique":
        try:
            x_inv = solve_inverse(A, b)
            x_cra, _ = solve_cramer(A, b)
            x_gau, _ = solve_gaussian(A, b)
            solution = x_gau
            c1, c2, c3 = st.columns(3)
            c1.metric("Inverse method", f"({x_inv[0]:.3f}, {x_inv[1]:.3f}, {x_inv[2]:.3f})")
            c2.metric("Cramer's rule", f"({x_cra[0]:.3f}, {x_cra[1]:.3f}, {x_cra[2]:.3f})")
            c3.metric("Gaussian elim.", f"({x_gau[0]:.3f}, {x_gau[1]:.3f}, {x_gau[2]:.3f})")
        except SingularMatrixError as exc:
            st.warning(f"Direct solve failed: {exc}")
    else:
        st.info("No single point solution — see the plane arrangement below.")

    st.plotly_chart(plane_figure(A, b, solution=solution), use_container_width=True)


# ============================================================================
# Page — Linear Algebra: Iterative Convergence
# ============================================================================

elif page == "LA — Iterative Convergence":
    import numpy as np
    import pandas as pd

    from matrix.presets import list_presets, get_preset, PRESETS
    from matrix.iterative_solvers import jacobi, gauss_seidel, sor, is_diagonally_dominant
    from matrix.plot_planes import convergence_figure, residual_figure
    from matrix.dashboard_helpers import system_from_editor

    st.header("Linear Algebra — Iterative Convergence")
    st.markdown(
        r"""
        Iterative methods start from a guess and **step toward** the solution.
        Watch each iterate walk through 3-space toward the intersection — and
        see how a non-diagonally-dominant matrix makes them diverge instead.
        """
    )

    preset = st.sidebar.selectbox(
        "Preset system", list_presets(),
        index=list_presets().index("Diagonally dominant (converges)"),
    )
    st.sidebar.caption(PRESETS[preset]["note"])
    method_name = st.sidebar.selectbox("Method", ["Jacobi", "Gauss-Seidel", "SOR"])
    max_iter = st.sidebar.slider("Iterations", min_value=5, max_value=100, value=25)
    omega = st.sidebar.slider("SOR ω", 0.5, 1.9, 1.1, 0.1) if method_name == "SOR" else 1.1

    A0, b0 = get_preset(preset)
    df = pd.DataFrame(np.column_stack([A0, b0]), columns=["x", "y", "z", "= b"])
    edited = st.data_editor(df, key=f"iter_{preset}", use_container_width=True)
    A, b = system_from_editor(edited)

    dominant = is_diagonally_dominant(A)
    st.write(f"Diagonally dominant: **{dominant}** "
             f"({'methods should converge' if dominant else 'methods may diverge'})")

    if method_name == "Jacobi":
        res = jacobi(A, b, max_iter=max_iter)
    elif method_name == "Gauss-Seidel":
        res = gauss_seidel(A, b, max_iter=max_iter)
    else:
        res = sor(A, b, omega=omega, max_iter=max_iter)

    st.write(
        f"**Converged: {res.converged}** after {res.iterations} iterations; "
        f"final residual = {res.residuals[-1]:.2e}"
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(convergence_figure(A, b, res.history), use_container_width=True)
    with col_b:
        st.plotly_chart(residual_figure(res.residuals), use_container_width=True)

    table = pd.DataFrame(res.history, columns=["x", "y", "z"])
    table["residual"] = res.residuals
    table.index.name = "iteration"
    st.dataframe(table, use_container_width=True)


# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>
    Built with Streamlit • NumPy • SciPy • Matplotlib
    </small>
    """,
    unsafe_allow_html=True,
)
