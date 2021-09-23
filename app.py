import pandas as pd
import streamlit as st

import altair as alt
import numpy as np
from scipy.stats import beta

# Sidebar
prior_success = st.sidebar.number_input("Beta Prior (Alpha)", 10)
prior_failure = st.sidebar.number_input("Beta Prior (Beta)", 10)

# AB test results
test_column, control_column = st.columns(2)
test_success = test_column.number_input("Test Success", 1, 1000)
test_failure = test_column.number_input("Test Failures", 1, 1000)
control_success = control_column.number_input("Control Success", 1, 1000)
control_failure = control_column.number_input("Control Failures", 1, 1000)

# Compute posterior distributions
alpha_test = prior_success + test_success
beta_test = prior_failure + test_failure
alpha_control = prior_success + control_success
beta_control = prior_failure + control_failure

x = np.linspace(0, 1, 1000)
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
    x="x",
    y="y",
    color="group"
))
