import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import norm


class OptionPricingEngine:
    """
    A class to represent an option pricing engine based on the Black-Scholes model.

    Attributes:
        S (float): The current stock price.
        K (float): The strike price of the option.
        T (float): The time to maturity in years.
        r (float): The risk-free interest rate.
        q (float): The dividend yield.
        option_type (str): Type of the option, either 'Call' or 'Put'.
        sigma (float): The implied volatility.
        price (float): The market price of the option.
        delta (float): The options delta.
        gamma (float): The options gamma.
        vega (float): The options vega.
        theta (float): The options theta.
        rho (float): The options rho.
    """

    def __init__(self, S, K, T, r, q, option_type="Call"):
        """
        Initialize the OptionPricingEngine with the given parameters.

        Parameters:
            S (float): The current stock price.
            K (float): The strike price of the option.
            T (float): The time to maturity in years.
            r (float): The risk-free interest rate.
            q (float): The dividend yield.
            option_type (str): The type of the option ("Call" or "Put").
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.option_type = option_type

    def set_implied_vol(self, market_price):
        """
        Set the implied volatility based on the market price of the option.

        Parameters:
            market_price (float): The market price of the option.
        """
        self.sigma = self._implied_volatility(market_price)
        self.pirce = self.calc_black_scholes(self.S)
        self._greeks_spot()

    def _implied_volatility(self, market_price):
        """
        Calculate the implied volatility using Brent's method.

        Parameters:
            market_price (float): The market price of the option.

        Returns:
            float: The implied volatility.
        """

        def objective_function(sigma):
            return self._black_scholes(sigma, self.S) - market_price

        try:
            return brentq(objective_function, 0.01, 2.0)
        except ValueError:
            return None

    def _d1_d2(self, sigma, S):
        """
        Calculate the d1 and d2 terms for the Black-Scholes model.

        Parameters:
            sigma (float): The volatility.
            S (float): The current stock price.

        Returns:
            tuple: A tuple of d1 and d2.
        """
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / (
            sigma * np.sqrt(self.T)
        )
        d2 = d1 - sigma * np.sqrt(self.T)
        return d1, d2

    def _black_scholes(self, sigma, S):
        """
        Calculate the Black-Scholes price for the option.

        Parameters:
            sigma (float): The volatility.
            S (float): The current stock price.

        Returns:
            float: The option price.
        """
        d1, d2 = self._d1_d2(sigma, S)

        if self.option_type == "Call":
            price = S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(
                -self.r * self.T
            ) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * np.exp(
                -self.q * self.T
            ) * norm.cdf(-d1)
        return price

    def calc_delta(self, S):
        """
        Calculate the option's Delta.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The option's Delta.
        """
        d1, d2 = self._d1_d2(self.sigma, S)
        delta = (
            np.exp(-self.q * self.T) * norm.cdf(d1)
            if self.option_type == "Call"
            else -np.exp(-self.q * self.T) * norm.cdf(-d1)
        )
        return delta

    def calc_gamma(self, S):
        """
        Calculate the option's Gamma.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The option's Gamma.
        """
        d1, d2 = self._d1_d2(self.sigma, S)
        gamma = (
            np.exp(-self.q * self.T) * norm.pdf(d1) / (S * self.sigma * np.sqrt(self.T))
        )
        return gamma

    def calc_vega(self, S):
        """
        Calculate the option's Vega.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The option's Vega.
        """
        d1, d2 = self._d1_d2(self.sigma, S)
        vega = S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
        return vega

    def calc_theta(self, S):
        """
        Calculate the option's Theta.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The option's Theta.
        """
        d1, d2 = self._d1_d2(self.sigma, S)
        theta = -(S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / (
            2 * np.sqrt(self.T)
        )
        return theta

    def calc_rho(self, S):
        """
        Calculate the option's Rho.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The option's Rho.
        """
        d1, d2 = self._d1_d2(self.sigma, S)
        rho = (
            self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
            if self.option_type == "Call"
            else -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        )
        return rho

    def _greeks_spot(self):
        """
        Calculate the option's Greeks.
        The Greeks include Delta, Gamma, Vega, Theta, and Rho.
        """
        self.delta = self.calc_delta(self.S)
        self.gamma = self.calc_gamma(self.S)
        self.vega = self.calc_vega(self.S)
        self.theta = self.calc_theta(self.S)
        self.rho = self.calc_rho(self.S)

    def calc_black_scholes(self, S):
        """
        Calculate the Black-Scholes option price for the given stock price.

        Parameters:
            S (float): The current stock price.

        Returns:
            float: The Black-Scholes option price.
        """
        return self._black_scholes(self.sigma, S)


def calculate_historic_vol(price_history):
    """
    Calculate the historical volatility of a stock based on price history.

    Parameters:
        price_history (DataFrame): A DataFrame containing historical stock prices.

    Returns:
        Series: A Pandas Series with the rolling volatility.
    """
    log_returns = np.log(price_history["S (EUR)"] / price_history["S (EUR)"].shift(-1))
    mean = log_returns.mean()
    std = log_returns.std()
    log_returns_filtered = log_returns.loc[(np.abs(log_returns) < mean + 4 * std)]

    return log_returns_filtered.rolling(window=30).std().dropna() * np.sqrt(256)


def compare_hist_vol(option, price_history):
    """
    Compare the current implied volatility with historical volatility.

    Parameters:
        option (OptionPricingEngine): The option pricing engine object.
        price_history (DataFrame): The historical price data.

    Returns:
        str: A string indicating whether the option is 'expensive', 'cheap', or 'fair'.
    """
    hist_vol = calculate_historic_vol(price_history)
    min_hist_vol = min(hist_vol)
    max_hist_vol = max(hist_vol)
    if option.sigma > max_hist_vol:
        return "expensive"
    elif option.sigma < min_hist_vol:
        return "cheap"
    else:
        return "fair"


def caluculate_time_to_maturity(data):
    """
    Calculate the time to maturity for an option.

    Parameters:
        data (DataFrame): A DataFrame containing option data, including 'Date' and 'Maturity'.

    Returns:
        float: The time to maturity in years.
    """
    delta_time = dt.datetime.strptime(data.Maturity, "%d/%m/%Y") - dt.datetime.strptime(
        data.Date, "%d/%m/%Y"
    )
    return delta_time.days / 365


def generate_data(outcome, options, price_history):
    """
    Generate data for various option Greeks and perceived value based on market prices.

    Parameters:
        outcome (str): The outcome to evaluate (e.g., 'Call' or 'Put').
        options (dict): A dictionary of options to evaluate.
        price_history (DataFrame): Historical price data to compare volatility.

    Returns:
        tuple: A tuple containing arrays of s_range, deltas, gammas, thetas, vegas, rhos, premium, perceived value, and implied volatility.
    """
    s_range = np.linspace(125, 185, 200)
    deltas = options[outcome].calc_delta(s_range)
    gammas = options[outcome].calc_gamma(s_range)
    thetas = options[outcome].calc_theta(s_range)
    vegas = options[outcome].calc_vega(s_range)
    rhos = options[outcome].calc_rho(s_range)
    premium = options[outcome].calc_black_scholes(s_range)
    percived_value = compare_hist_vol(options[outcome], price_history)
    implied_vol = options[outcome].sigma
    return (
        s_range,
        deltas,
        gammas,
        thetas,
        vegas,
        rhos,
        premium,
        percived_value,
        implied_vol,
    )
