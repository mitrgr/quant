import os

import pandas as pd
import scipy
import streamlit as st
from compute_quant_tasks import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

options_data = pd.read_csv(os.path.join(DATA_DIR, "options.csv"), sep=";")
price_hist = pd.read_csv(os.path.join(DATA_DIR, "price_hist.csv"), sep=";")


hist_vol = calculate_historic_vol(price_hist)

options_data["time_to_maturity"] = options_data.apply(
    caluculate_time_to_maturity, axis=1
)

risk_free_rate = 0.02
dividend_yield = 0.03
spot = price_hist["S (EUR)"].iloc[0]

options = dict()
for index, option in options_data.iterrows():
    options[option.Name] = OptionPricingEngine(
        S=spot,
        K=option.Strike,
        T=option.time_to_maturity,
        r=risk_free_rate,
        q=dividend_yield,
        option_type=option.Type,
    )
    options[option.Name].set_implied_vol(option.Premium)

option_selecter = options_data.Name

# Streamlit app
st.set_page_config(layout="wide")
st.title("Dashboard: Risk management")

selected_outcome = st.selectbox("Select an option", option_selecter)
(
    s_range,
    y_deltas,
    y_gammas,
    y_thetas,
    y_vegas,
    y_rhos,
    y_premium,
    percived_value,
    implied_vol,
) = generate_data(selected_outcome, options, price_hist)

st.header(f"{selected_outcome} is {percived_value}")
data = {
    "historic volatility (30d)": hist_vol,
    f"Implied volatility": implied_vol,
}

st.line_chart(data, x_label=f"time", y_label="Volatility", color=["#FF0000", "#0000FF"])


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Delta", "Gamma", "Theta", "Vega", "Rho", "Black Scholes"]
)

with tab1:
    st.write(f"### Delta for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].delta}")
    data = {"s_range": s_range, "y_deltas": y_deltas}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="Delta")


with tab2:
    st.write(f"### Gamma for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].gamma}")
    data = {"s_range": s_range, "y_gammas": y_gammas}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="Gamma")


with tab3:
    st.write(f"### Theta for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].theta}")
    data = {"s_range": s_range, "y_thetas": y_thetas}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="Theta")


with tab4:
    st.write(f"### Vegas for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].vega}")
    data = {"s_range": s_range, "y_vegas": y_vegas}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="Vega")


with tab5:
    st.write(f"### Rho for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].rho}")
    data = {"s_range": s_range, "y_rhos": y_rhos}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="rho")

with tab6:
    st.write(f"### Black scholes for {selected_outcome}")
    st.write(f"####  {options[selected_outcome].pirce}")
    data = {"s_range": s_range, "y_premium": y_premium}

    df = pd.DataFrame(data)
    st.line_chart(df.set_index("s_range"), x_label=f"S", y_label="Premium")
