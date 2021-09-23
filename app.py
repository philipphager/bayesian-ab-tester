import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta

# Sidebar
prior_success = st.sidebar.number_input("Beta Prior (Alpha)", 10)
prior_failure = st.sidebar.number_input("Beta Prior (Beta)", 10)

# AB test results
test_column, control_column = st.columns(2)
test_success = test_column.number_input("Test Success", 1, 100_000)
test_failure = test_column.number_input("Test Failures", 1, 100_000)
control_success = control_column.number_input("Control Success", 1, 100_000)
control_failure = control_column.number_input("Control Failures", 1, 100_000)

# Compute posterior distributions
alpha_test = prior_success + test_success
beta_test = prior_failure + test_failure
alpha_control = prior_success + control_success
beta_control = prior_failure + control_failure

x = np.linspace(0, 1, 10_000)
y_test = beta.pdf(x, alpha_test, beta_test)
y_control = beta.pdf(x, alpha_control, beta_control)

# Plot
source = pd.DataFrame({
    "x": x,
    "y_test": y_test,
    "y_control": y_control
})

source = source.melt(id_vars="x", var_name="group", value_name="y")

st.write(alt.Chart(source, width=700, height=400).mark_area(opacity=0.75).encode(
    x=alt.X("x", title=""),
    y=alt.Y("y", title="Probability Density Function"),
    color="group"
))

samples_test = beta.rvs(alpha_test, beta_test, size=1_000_000)
samples_control = beta.rvs(alpha_control, beta_control, size=1_000_000)
probability_uplift = np.mean(samples_test > samples_control)
samples_uplift = (samples_test - samples_control)

x = np.arange(-1, 1.025, 0.025)
y, _ = np.histogram(samples_uplift, bins=x, density=True)

source = pd.DataFrame({"x": x[:-1], "y": y})

title = f"Uplift of Test over Control {probability_uplift:.2f}"
st.write(alt.Chart(source, title=title, width=620, height=300).mark_bar(opacity=0.75).encode(
    x="x",
    y=alt.Y("y", title="Probability Density Function"),
    color=alt.condition(alt.datum.x < 0, alt.value("#d62728"), alt.value("#2ca02c")),
) + alt.Chart(pd.DataFrame({"x": [0]}), title=title, width=620, height=300).mark_rule(size=2).encode(
    x="x",
    color=alt.value("#000000"),
))
