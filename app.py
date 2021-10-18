import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta, norm

from utils import log_beta_mean, var_beta_mean, beta_gq


def plot_prior(prior_alpha, prior_beta):
    x = np.linspace(0, 1, 10_000)
    y = beta.pdf(x, prior_alpha, prior_beta)

    source = pd.DataFrame({
        "x": x,
        "y": y,
    })

    title = "Prior Conversion Rate"
    return alt.Chart(source, title=title, width=630, height=300).mark_area(opacity=0.75).encode(
        x=alt.X("x", title=""),
        y=alt.Y("y", title="Probability Density Function"),
    ).interactive(bind_y=False)


def plot_posteriors(alpha_test, beta_test, alpha_control, beta_control):
    x = np.linspace(0, 1, 10_000)
    y_test = beta.pdf(x, alpha_test, beta_test)
    y_control = beta.pdf(x, alpha_control, beta_control)

    source = pd.DataFrame({
        "x": x,
        "test": y_test,
        "control": y_control
    })

    source = source.melt(id_vars="x", var_name="group", value_name="y")
    title = "Posterior Conversion Rates"

    return alt.Chart(source, title=title, width=700, height=300).mark_area(opacity=0.75).encode(
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
    return alt.Chart(source, title=title, width=630, height=300).mark_bar(opacity=0.75).encode(
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
is_local_prior = st.sidebar.checkbox("Set prior per metric", False)
st.sidebar.markdown("___")

if not is_local_prior:
    prior_success = st.sidebar.number_input("Global Prior - Alpha", min_value=1, value=10)
    prior_failure = st.sidebar.number_input("Global Prior - Beta", min_value=1, value=10)
    st.sidebar.markdown("___")

credibility = st.sidebar.number_input("Credibility Interval", min_value=1, max_value=100, value=95)

columns = ["metric", "group", "success", "total"]
file = st.file_uploader("Upload a .CSV file with columns:" + ", ".join(columns))

if file is not None:
    df = pd.read_csv(file)

    for column in columns:
        if column not in df.columns:
            st.error(f"CSV file misses column: {column}")
            st.stop()

    control_df = df[df["group"] == "control"]
    test_df = df[df["group"] != "control"]

    if len(control_df) != len(test_df):
        st.error("""
        Number of control and test groups do not match.
        Ensure to have two rows per metric, one with a group name 'control'.
        """)
        st.stop()

    df = control_df.merge(test_df, on=["metric"], suffixes=("_control", "_test"))
    df = df.sort_values("total_test", ascending=False)
else:
    st.stop()

for i, row in df.iterrows():
    st.subheader(row.metric)

    if is_local_prior:
        c1, c2 = st.columns(2)
        prior_success = c1.number_input("Prior - Alpha", min_value=1, value=10, key=row.metric)
        prior_failure = c2.number_input("Prior - Beta", min_value=1, value=10, key=row.metric)
        st.markdown("")

    test_success = row.success_test
    test_failure = row.total_test - row.success_test
    control_success = row.success_control
    control_failure = row.total_control - row.success_control

    # Compute posterior distributions
    alpha_test = prior_success + test_success
    beta_test = prior_failure + test_failure
    alpha_control = prior_success + control_success
    beta_control = prior_failure + control_failure

    st.write(plot_prior(prior_success, prior_failure))
    st.write(plot_posteriors(alpha_test, beta_test, alpha_control, beta_control))
    st.write(plot_uplift(alpha_test, beta_test, alpha_control, beta_control))

    users = 100
    test_probability = get_test_probability(alpha_test, beta_test, alpha_control, beta_control)
    credibility_lower, credibility_upper = get_credibility_interval(alpha_test, beta_test, alpha_control, beta_control,
                                                                    credibility)
    risk_control, risk_test = get_risk(alpha_test, beta_test, alpha_control, beta_control, users)

    st.markdown(f"Test Success:`{test_success} / {row.total_test} = {(test_success / row.total_test * 100):.2f}%`")
    st.markdown(f"Control Success:`{control_success} / {row.total_control} = {(control_success / row.total_control * 100):.2f}%`")
    st.markdown(f"Probability of conversions improving in Test: `{test_probability:.2f}%`")
    st.markdown(f"{credibility}% credibility interval: `[{credibility_lower:.2f}%, {credibility_upper:.2f}%]`")
    st.markdown(f"Risk of choosing Control is loosing `{risk_control}` conversions per `{users}` users")
    st.markdown(f"Risk of choosing Test is loosing `{risk_test}` conversions per `{users}` users")
    st.markdown("""
    ---
    """)
