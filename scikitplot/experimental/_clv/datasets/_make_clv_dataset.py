import numpy as np
import pandas as pd


def make_clv_dataset(model: str, size=1, **kwargs) -> pd.DataFrame:
    """
    Generate synthetic customer lifetime value (CLV) data for various models including:
    - Beta-Geometric/Beta-Binomial (BG/BB)
    - MBG/NBD
    - Pareto/NBD
    - BG/NBD

    Parameters
    ----------
    model : str
        The type of model to use for generating data. Must be one of 'bg_nbd', 'pareto_nbd',
        'mbg_nbd', or 'bg_bb'.
    size : int, optional
        The number of customers to generate. Default is 1.
    **kwargs : dict
        Additional parameters specific to the chosen model, including but not limited to:
        - T : array_like
            The length of time observing new customers.
        - r, alpha, a, b, s, beta : float
            Model-specific parameters.
        - observation_period_end : str, optional
            The date observation ends (for transactional data).
        - freq : str, optional
            Frequency of transactions (e.g., 'D' for days).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing generated customer data with relevant columns depending on the model.

    Raises
    ------
    ValueError
        If an invalid model type is provided.

    References
    ----------
    .. [1]:
    .. [2]:
    .. [3]:
    .. [4]:

    Examples
    --------
    >>> df = make_clv_dataset(
    ...     'bg_nbd',
    ...     size=1000,
    ...     T=10,
    ...     r=0.5,
    ...     alpha=1.0,
    ...     a=0.5,
    ...     b=0.5,
    ... )
    >>> df.head()
       customer_id  frequency  recency   T   lambda     p  alive
    0            0          3      5.5  10  0.4325  0.57      1
    1            1          2      7.2  10  0.5234  0.62      0

    """
    # Dispatch to the appropriate model function
    model_funcs = {
        "bg_nbd": beta_geometric_nbd_model,
        "pareto_nbd": pareto_nbd_model,
        "mbg_nbd": modified_beta_geometric_nbd_model,
        "bg_bb": beta_geometric_beta_binom_model,
    }

    if model not in model_funcs:
        raise ValueError(
            f"Invalid model type: {model}. Must be one of {list(model_funcs.keys())}."
        )

    model_func = model_funcs[model]
    return model_func(size=size, **kwargs)


def beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate customer data using the Beta-Geometric NBD model.

    Parameters
    ----------
    T : array_like
        The length of time observing new customers.
    r, alpha, a, b : float
        Model-specific parameters.
    size : int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated customer data, including frequency, recency, etc.

    """
    # Ensure T is an array of the correct size
    T = np.full(size, T) if isinstance(T, (int, float)) else np.asarray(T)

    probability_of_post_purchase_death = np.random.beta(a, b, size=size)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    # Prepare DataFrame
    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # Simulate purchase events
        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        alive = True
        while np.sum(times) + next_purchase_in < T[i] and alive:
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)
            alive = np.random.uniform() > p

        times = np.array(times).cumsum()
        df.iloc[i] = (
            len(np.unique(times.astype(int))),
            np.max(times if len(times) > 0 else 0),
            T[i],
            l,
            p,
            alive,
            i,
        )

    return df.set_index("customer_id")


def pareto_nbd_model(T, r, alpha, s, beta, size=1):
    """
    Generate customer data using the Pareto-NBD model.

    Parameters
    ----------
    T : array_like
        The length of time observing new customers.
    r, alpha, s, beta : float
        Model-specific parameters.
    size : int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated customer data, including frequency, recency, etc.

    """
    T = np.full(size, T) if isinstance(T, (int, float)) else np.asarray(T)

    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)
    mus = np.random.gamma(s, scale=1.0 / beta, size=size)

    columns = ["frequency", "recency", "T", "lambda", "mu", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        l = lambda_[i]
        mu = mus[i]
        time_of_death = np.random.exponential(scale=1.0 / mu)

        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        while np.sum(times) + next_purchase_in < min(time_of_death, T[i]):
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)

        times = np.array(times).cumsum()
        df.iloc[i] = (
            len(np.unique(times.astype(int))),
            np.max(times if len(times) > 0 else 0),
            T[i],
            l,
            mu,
            time_of_death > T[i],
            i,
        )

    return df.set_index("customer_id")


def modified_beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate customer data using the Modified Beta-Geometric NBD model.

    Parameters
    ----------
    T : array_like
        The length of time observing new customers.
    r, alpha, a, b : float
        Model-specific parameters.
    size : int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated customer data, including frequency, recency, etc.

    """
    T = np.full(size, T) if isinstance(T, (int, float)) else np.asarray(T)

    probability_of_post_purchase_death = np.random.beta(a, b, size=size)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        alive = np.random.uniform() > p
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)
            alive = np.random.uniform() > p

        times = np.array(times).cumsum()
        df.iloc[i] = (
            len(np.unique(times.astype(int))),
            np.max(times if len(times) > 0 else 0),
            T[i],
            l,
            p,
            alive,
            i,
        )

    return df.set_index("customer_id")


def beta_geometric_beta_binom_model(N, alpha, beta, gamma, delta, size=1):
    """
    Generate customer data using the Beta-Geometric/Beta-Binomial model.

    Parameters
    ----------
    N : array_like
        The total number of periods for each customer.
    alpha, beta, gamma, delta : float
        Model-specific parameters.
    size : int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated customer data, including frequency, recency, etc.

    """
    N = np.full(size, N) if isinstance(N, (int, float)) else np.asarray(N)

    probability_of_post_purchase_death = np.random.beta(a=alpha, b=beta, size=size)
    thetas = np.random.beta(a=gamma, b=delta, size=size)

    columns = [
        "frequency",
        "recency",
        "n_periods",
        "p",
        "theta",
        "alive",
        "customer_id",
    ]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)
    for i in range(size):
        p = probability_of_post_purchase_death[i]
        theta = thetas[i]

        current_t = 0
        alive = True
        times = []
        while current_t < N[i] and alive:
            alive = np.random.binomial(1, theta) == 0
            if alive and np.random.binomial(1, p) == 1:
                times.append(current_t)
            current_t += 1

        if alive:
            alive = np.random.binomial(1, theta) == 0

        df.iloc[i] = (
            len(times),
            times[-1] + 1 if len(times) != 0 else 0,
            N[i],
            p,
            theta,
            alive,
            i,
        )

    return df
