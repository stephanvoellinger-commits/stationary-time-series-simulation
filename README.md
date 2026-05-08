# Stationary Time Series Simulation

Simulation framework for stationary stochastic processes constructed from transformed Brownian motion and time-dependent diffeomorphic mappings.

## Overview

This project investigates the construction of stationary stochastic processes with controllable covariance structures using transformed Brownian motion and time-change techniques.

The implementation includes:

- Monte Carlo simulation
- Empirical autocorrelation analysis
- Interactive parameter control
- Stationary distribution analysis
- Visualization of stochastic sample paths

## Mathematical Background

The project is motivated by Ornstein-Uhlenbeck-type dynamics and generalized stationary diffusion processes.

Example reference model:

\[
dX_t = -\alpha X_t\,dt + \sigma\,dW_t
\]

The framework explores generalized transformations of Brownian motion of the form

\[
Y_t = \Phi_t(W_{\tau(t)})
\]

with the goal of constructing stationary processes with prescribed structural properties.

## Features

- Interactive simulation environment
- Adjustable autocorrelation parameter
- Visualization of stationary behavior
- Numerical experiments for covariance structures
- Python-based stochastic simulation framework

## Technologies

- Python
- NumPy
- Matplotlib

## Usage

```bash
python Pythonstationaryprocess.py
```

## Motivation

The project was developed as an independent quantitative research and simulation project in stochastic processes and time series analysis.
