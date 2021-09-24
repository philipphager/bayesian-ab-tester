import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta, norm

from utils import log_beta_mean, var_beta_mean, beta_gq


def plot_posteriors(alpha_test, beta_test, alpha_control, beta_control):
    x = np.linspace(0, 1, 10_000)
    y_test = beta.pdf(x, alpha_test, beta_test)
    y_control = beta.pdf(x, alpha_control, beta_control)

    source = pd.DataFrame({
        "x": x,
        "y_test": y_test,
        "y_control": y_control
    })

    source = source.melt(id_vars="x", var_name="group", value_name="y")

    return alt.Chart(source, width=700, height=300).mark_area(opacity=0.75).encode(
        x=alt.X("x", title=""),
        y=alt.Y("y", title="Probability Density Function"),
        color="group"
    ).interactive(bind_y=False)


def plot_uplift(alpha_test, beta_test, alpha_control, beta_control, samples=1_000_000):
    samples_test = beta.rvs(alpha_test, beta_test, size=samples)
    samples_control = beta.rvs(alpha_control, beta_control, size=samples)
    samples_uplift = (samples_test - samples_control) / samples_control

    # Compute PDF of samples
    x = np.arange(-1.0, 1.05, 0.025)
    y, _ = np.histogram(samples_uplift, bins=x, density=True)
    source = pd.DataFrame({"x": x[:-1], "y": y})

    title = ""
    return alt.Chart(source, title=title, width=620, height=300).mark_bar(opacity=0.75).encode(
        x="x",
        y=alt.Y("y", title="Probability Density Function"),
        color=alt.condition(alt.datum.x < 0, alt.value("#d62728"), alt.value("#2ca02c")),
    ) + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(size=2).encode(
        x=alt.X("x", title=""),
        color=alt.value("#000000"),
    ).interactive(bind_y=False)


def get_test_probability(alpha_test, beta_test, alpha_control, beta_control):
    return norm(
        loc=beta.mean(alpha_test, beta_test) - beta.mean(alpha_control, beta_control),
        scale=np.sqrt(beta.var(alpha_test, beta_test) + beta.var(alpha_control, beta_control))
    ).sf(0) * 100


def get_credibility_interval(alpha_test, beta_test, alpha_control, beta_control, probability):
    d2_beta = norm(
        loc=log_beta_mean(alpha_test, beta_test) - log_beta_mean(alpha_control, beta_control),
        scale=np.sqrt(var_beta_mean(alpha_test, beta_test) + var_beta_mean(alpha_control, beta_control))
    )

    lower = (1 - probability / 100) / 2
    upper = (1 - lower)
    return (np.exp(d2_beta.ppf((lower, upper))) - 1) * 100


def get_risk(alpha_test, beta_test, alpha_control, beta_control, users):
    nodes_control, weights_control = beta_gq(24, alpha_control, beta_control)
    nodes_test, weights_test = beta_gq(24, alpha_test, beta_test)

    gq = sum(nodes_control * beta.cdf(nodes_control, alpha_test, beta_test) * weights_control) + \
         sum(nodes_test * beta.cdf(nodes_test, alpha_control, beta_control) * weights_test)
    risk_beta = gq - beta.mean((alpha_control, alpha_test), (beta_control, beta_test))

    risk_control = risk_beta[0] * users
    risk_test = risk_beta[1] * users
    return round(risk_control, 2), round(risk_test, 2)


# Sidebar
prior_success = st.sidebar.number_input("Beta Prior - Alpha", min_value=1, value=10)
prior_failure = st.sidebar.number_input("Beta Prior - Beta", min_value=1, value=10)
credibility = st.sidebar.number_input("Credibility Interval", min_value=1, max_value=100, value=95)

# AB test results
column_1, column_2 = st.columns(2)
test_success = column_1.number_input("Test Success", min_value=1, value=51)
test_total = column_2.number_input("Test Total", min_value=1, value=100)
test_failure = test_total - test_success

control_success = column_1.number_input("Control Success", min_value=1, value=49)
control_total = column_2.number_input("Control Total", min_value=1, value=100)
control_failure = control_total - control_success

st.markdown("___")

if control_success > control_total or test_success > test_total:
    st.error("The number of successful users must be less or equal to your total users")
    st.stop()

# Compute posterior distributions
alpha_test = prior_success + test_success
beta_test = prior_failure + test_failure
alpha_control = prior_success + control_success
beta_control = prior_failure + control_failure

st.write(plot_posteriors(alpha_test, beta_test, alpha_control, beta_control))
st.write(plot_uplift(alpha_test, beta_test, alpha_control, beta_control))

users = 100
test_probability = get_test_probability(alpha_test, beta_test, alpha_control, beta_control)
credibility_lower, credibility_upper = get_credibility_interval(alpha_test, beta_test, alpha_control, beta_control,
                                                                credibility)
risk_control, risk_test = get_risk(alpha_test, beta_test, alpha_control, beta_control, users)

st.markdown(f"Probability of Conversion improving in Test: `{test_probability:.2f}%`")
st.markdown(f"{credibility}% credibility interval: `[{credibility_lower:.2f}%, {credibility_upper:.2f}%]`")
st.markdown(f"Risk of choosing Control is loosing `{risk_control}` conversions per `{users}` users")
st.markdown(f"Risk of choosing Test is loosing `{risk_test}` conversions per `{users}` users")
