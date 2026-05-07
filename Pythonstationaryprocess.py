import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# ============================================================
# PARAMETERS
# ============================================================

T = 10.0
dt = 0.01
alpha_init = 0.5


# ============================================================
# sigma(alpha)
# ============================================================

def sigma_from_alpha(alpha):

    alpha = np.clip(alpha, 1e-6, 0.999999)

    return 1 / np.sqrt(-2 * np.log(alpha))


# ============================================================
# Brownian Motion
# ============================================================

def simulate_bm_zero(T, dt):

    n = int(T / dt)

    t = np.linspace(0, T, n)

    X = np.zeros(n)

    for i in range(1, n):

        X[i] = X[i-1] + np.sqrt(dt) * np.random.randn()

    return t, X


# ============================================================
# TRANSFORMED PROCESS
# ============================================================

def simulate_transformed(T, dt, alpha):

    n = int(T / dt)

    t = np.linspace(0, T, n)

    sigma = sigma_from_alpha(alpha)

    Y = np.zeros(n)

    dY = np.zeros(n)

    dX = np.zeros(n)

    # stationary initial distribution
    Y[0] = np.random.normal(0, sigma)

    # Brownian increments
    for i in range(1, n):

        dX[i] = np.sqrt(dt) * np.random.randn()

    # transformed increments
    for i in range(1, n):

        dY[i] = np.exp(np.log(alpha) * i * -1 * dt) * dX[i]

    # cumulative sum
    for i in range(1, n):

        Y[i] = Y[i-1] + dY[i]

    # final exponential transformation
    for i in range(n):

        Y[i] = np.exp(np.log(alpha) * i * dt) * Y[i]

    # normalization
    return t, Y * (1 / sigma_from_alpha(alpha))


# ============================================================
# MONTE CARLO VERSION
# ============================================================

def simulate_mc(n_paths, T, dt, alpha):

    n = int(T / dt)

    sigma = sigma_from_alpha(alpha)

    Y = np.zeros((n_paths, n))

    dY = np.zeros((n_paths, n))

    dX = np.zeros((n_paths, n))

    # stationary initial distribution
    Y[:, 0] = np.random.normal(0, sigma, size=n_paths)

    # Brownian increments
    for i in range(1, n):

        dX[:, i] = np.sqrt(dt) * np.random.randn(n_paths)

    # transformed increments
    for i in range(1, n):

        dY[:, i] = (
            np.exp(np.log(alpha) * i * -1 * dt)
            * dX[:, i]
        )

    # cumulative sum
    for i in range(1, n):

        Y[:, i] = Y[:, i-1] + dY[:, i]

    # final exponential transformation
    for i in range(n):

        Y[:, i] = (
            np.exp(np.log(alpha) * i * dt)
            * Y[:, i]
        )

    # normalization
    Y = Y * (1 / sigma_from_alpha(alpha))

    return Y


# ============================================================
# EMPIRICAL ACF
# ============================================================

def empirical_acf_10(Y, dt):

    n_paths, n = Y.shape

    # center process
    Y_centered = Y - np.mean(Y, axis=0)

    var0 = np.mean(Y_centered[:, 0] ** 2)

    times = np.linspace(0, 10, 11)

    indices = np.clip(
        (times / dt).astype(int),
        0,
        n - 1
    )

    acf = []

    for idx in indices:

        cov = np.mean(
            Y_centered[:, 0] * Y_centered[:, idx]
        )

        acf.append(cov / var0)

    return times, np.array(acf)


# ============================================================
# INITIAL SIMULATION
# ============================================================

t_vals, X_path = simulate_bm_zero(T, dt)

_, Y_path = simulate_transformed(
    T,
    dt,
    alpha_init
)


# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(18, 8)
)

plt.subplots_adjust(bottom=0.4)

fig.suptitle(
    "Stationary Transformation of Brownian Motion",
    fontsize=14
)

# Brownian motion
line1, = ax1.plot(t_vals, X_path)

ax1.set_title("Brownian Motion")

ax1.set_ylim(-4, 4)

# transformed process
line2, = ax2.plot(t_vals, Y_path)

ax2.set_title(
    "Transformed Process"
)

ax2.set_ylim(-4, 4)


# ============================================================
# SLIDER
# ============================================================

ax_alpha = plt.axes([0.1, 0.25, 0.8, 0.03])

slider_alpha = Slider(
    ax_alpha,
    "alpha",
    0.05,
    0.99999,
    valinit=alpha_init
)


# ============================================================
# BUTTONS
# ============================================================

ax_new = plt.axes([0.1, 0.15, 0.2, 0.06])

button_new = Button(ax_new, "New Path")

ax_hist = plt.axes([0.35, 0.15, 0.2, 0.06])

button_hist = Button(ax_hist, "MC Histogram")

ax_acf = plt.axes([0.6, 0.15, 0.2, 0.06])

button_acf = Button(ax_acf, "MC ACF")


# ============================================================
# TEXTBOX
# ============================================================

ax_text = plt.axes([0.38, 0.05, 0.18, 0.06])

textbox = TextBox(
    ax_text,
    "MC paths",
    initial="20000"
)


# ============================================================
# INFO BUTTON
# ============================================================

ax_info = plt.axes([0.57, 0.05, 0.08, 0.06])

button_info = Button(ax_info, "Info")


# ============================================================
# UPDATE
# ============================================================

def update(val):

    global X_path, Y_path

    alpha = slider_alpha.val

    t_vals, X_path = simulate_bm_zero(T, dt)

    _, Y_path = simulate_transformed(
        T,
        dt,
        alpha
    )

    line1.set_ydata(X_path)

    line2.set_ydata(Y_path)

    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw_idle()


slider_alpha.on_changed(update)


# ============================================================
# NEW PATH
# ============================================================

def new_path(event):

    update(None)


button_new.on_clicked(new_path)


# ============================================================
# HISTOGRAM
# ============================================================

def run_hist(event):

    alpha = slider_alpha.val

    try:
        n_paths = int(textbox.text)

    except:
        n_paths = 20000

    Y = simulate_mc(
        n_paths,
        T,
        dt,
        alpha
    )

    # final time distribution
    data = Y[:, -1]

    fig2, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        data,
        bins=50,
        density=True
    )

    ax.set_title(
        "Monte Carlo Histogram of Y(T) \n"
        "Without restriction t=T, stationarity for all times"
    )

    ax.set_xlabel("Y(T)")

    ax.set_ylabel("Density")

    plt.show()


button_hist.on_clicked(run_hist)


# ============================================================
# ACF
# ============================================================

def run_acf(event):

    alpha = slider_alpha.val

    try:
        n_paths = int(textbox.text)

    except:
        n_paths = 20000

    Y = simulate_mc(
        n_paths,
        T,
        dt,
        alpha
    )

    times, acf_vals = empirical_acf_10(Y, dt)

    # theoretical ACF
    theory = alpha ** times

    fig3, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        times,
        acf_vals,
        marker='o',
        label='Monte Carlo ACF'
    )

    ax.plot(
        times,
        theory,
        linestyle='--',
        label='Theory'
    )

    ax.set_title(
        "Autocorrelation Function"
    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Correlation")

    ax.legend()

    plt.show()


button_acf.on_clicked(run_acf)


# ============================================================
# INFO BOX
# ============================================================

def show_info(event):

    fig_info, ax = plt.subplots(figsize=(8, 6))

    ax.axis("off")

    text = (
        "Monte Carlo Simulation (MC):\n"
        "- Simulates many transformed paths Y_t\n"
        "- Histogram approximates the distribution of Y(T)\n\n"
        "Autocorrelation Function (ACF):\n"
        "- Measures temporal dependence of Y_t\n"
        "- Compared with theoretical alpha^t\n"
    )

    ax.text(
        0,
        1,
        text,
        va="top",
        fontsize=11
    )

    plt.show()


button_info.on_clicked(show_info)


# ============================================================
# SHOW
# ============================================================

plt.show()