"""
Distributions for the CLV module.

BetaGeoBetaBinom(name, *args, **kwargs): Population-level distribution class for a discrete, non-contractual, Beta-Geometric/Beta-Binomial process.

BetaGeoBetaBinomRV([name, ndim_supp, ...])

ContContract(name, *args[, rng, dims, ...]): Distribution class of a continuous contractual data-generating process.

ContContractRV([name, ndim_supp, ...])

ContNonContract(name, *args[, rng, dims, ...]): Individual-level model for the customer lifetime value.

ContNonContractRV([name, ndim_supp, ...])

ParetoNBD(name, *args[, rng, dims, initval, ...]): Population-level distribution class for a continuous, non-contractual, Pareto/NBD process.

ParetoNBDRV([name, ndim_supp, ndims_params, ...])
"""

import numpy as np
import pandas as pd

__all__ = [
    "beta_geometric_beta_binom_model",
    "beta_geometric_nbd_model",
    "beta_geometric_nbd_model_transactional_data",
    "make_clv_dataset",
    "modified_beta_geometric_nbd_model",
    "pareto_nbd_model",
]


def make_clv_dataset(model: str, size=1, **kwargs) -> pd.DataFrame:
    """
    Generate synthetic customer lifetime value (CLV) data for various models including:

    * BG/BB      : BetaGeoBetaBinom Beta-Geometric/Beta-Binomial model data [1]_
    * MBG/NBD    : Modified Beta-Geometric/NBD model data [2]_
    * Pareto/NBD : ParetoNBD model data [3]_
    * BG/NBD     : Beta-Geometric/NBD model data [4]_
    * ... Additional models [5]_.

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
    .. [1]: "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)
    .. [2]: Batislam, E.P., M. Denizel, A. Filiztekin (2007), "Empirical validation and comparison of models for customer base analysis," International Journal of Research in Marketing, 24 (3), 201-209.
    .. [3]: Fader, Peter S. and Bruce G. S. Hardie (2005), "A Note on Deriving the Pareto/NBD Model and Related Expressions," <http://brucehardie.com/notes/009/>.
    .. [4]: "Customer-Base Analysis in a Discrete-Time Noncontractual Setting," Marketing Science, 29 (6), 1086-1108.

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
    Generate artificial data according to the population-level distribution class
    for a discrete, non-contractual, Beta-Geometric/Beta-Binomial process.

    The BG/NBD model simulates customer behavior over time, including
    purchase frequency and recency, as well as the probability of customer
    death (discontinuation of purchases).

    See [1] for model details

    Parameters
    ----------
    T: array_like
        The length of time observing new customers. If a scalar is provided,
        all customers will be observed for the same period.
    r, alpha, a, b: float
        Parameters for the BG/NBD model. 'r' and 'alpha' relate to the
        Poisson-gamma distribution for purchase rates, and 'a' and 'b'
        are the parameters for the Beta distribution for customer death. See [1]_
    size: int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pandas.DataFrame
        DataFrame with customer-level data, indexed by customer_id and containing
        'frequency' (number of purchases), 'recency' (time of last purchase),
        'T' (observation period), 'lambda' (purchase rate), 'p' (probability of death),
        'alive' (whether the customer is alive), and 'customer_id' (ID of the customer).


    References
    ----------
    .. [1]: Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
        "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
        Marketing Science, 29 (6), 1086-1108.
        https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf

    """
    # Ensure T is an array of the same size as the number of customers
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    # Generate random probabilities of post-purchase death (Beta distribution)
    probability_of_post_purchase_death = np.random.beta(a, b, size=size)

    # Generate random purchase rate lambdas (Gamma distribution)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    # Initialize columns for the resulting DataFrame
    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    # Loop over each customer to simulate their purchase behavior
    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        # Simulate times between purchases (exponentially distributed)
        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        alive = True

        # Simulate the purchase process until the customer is either dead or the observation period ends
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)
            alive = np.random.random() > p  # Customer death happens with probability p

        # Assign simulated data to the DataFrame
        times = np.array(
            times
        ).cumsum()  # Cumulative sum to get total time until each purchase
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[
                0
            ],  # Number of distinct purchases
            np.max(times if times.shape[0] > 0 else 0),  # Recency (last purchase time)
            T[i],  # Total observation period
            l,  # Purchase rate (lambda)
            p,  # Probability of death
            alive,  # Whether the customer is alive
            i,  # Customer ID
        )

    return df.set_index(
        "customer_id"
    )  # Return the DataFrame with customer IDs as the index


def beta_geometric_nbd_model_transactional_data(
    T, r, alpha, a, b, observation_period_end="2019-1-1", freq="D", size=1
):
    """
    Generate artificial transactional data according to the BG/NBD ... model.

    This function simulates the individual transactions of customers, not just
    their aggregate behavior. The simulated transactions include the date and
    customer information.

    See [1] for model details

    Parameters
    ----------
    T: int, float or array_like
        The length of time observing new customers. If a scalar is provided,
        all customers will be observed for the same period.
    r, alpha, a, b: float
        Parameters for the BG/NBD model, as described in the first function. See [1]_
    observation_period_end: date_like
        The date the observation period ends. The observation starts from the
        calculated start date for each customer.
    freq: string, optional
        The frequency of transactions. Default is 'D' for days, but could be 'W' for weeks, 'h' for hours, etc.
    size: int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with 'customer_id' and 'date' columns, representing individual
        transactions for each customer.

    References
    ----------
    .. [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)

    """
    observation_period_end = pd.to_datetime(
        observation_period_end
    )  # Convert observation end date to datetime

    # Handle the case where T is a scalar or an array
    if type(T) in [float, int]:
        start_date = [observation_period_end - pd.Timedelta(T - 1, unit=freq)] * size
        T = T * np.ones(size)
    else:
        start_date = [
            observation_period_end - pd.Timedelta(T[i] - 1, unit=freq)
            for i in range(size)
        ]
        T = np.asarray(T)

    # Generate random probabilities of post-purchase death (Beta distribution)
    probability_of_post_purchase_death = np.random.beta(a, b, size=size)

    # Generate random purchase rate lambdas (Gamma distribution)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    # Initialize columns for the resulting DataFrame
    columns = ["customer_id", "date"]
    df = pd.DataFrame(columns=columns)

    # Simulate the transactions for each customer
    for i in range(size):
        s = start_date[i]
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]
        age = T[i]

        # Start the list of purchases with the initial time (0th purchase)
        purchases = [[i, s - pd.Timedelta(1, unit=freq)]]
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        alive = True

        # Simulate subsequent purchases until the customer is dead or the observation period ends
        while next_purchase_in < age and alive:
            purchases.append([i, s + pd.Timedelta(next_purchase_in, unit=freq)])
            next_purchase_in += np.random.exponential(scale=1.0 / l)
            alive = np.random.random() > p  # Customer death with probability p

        # Append the customer's transactions to the DataFrame
        df = df.append(pd.DataFrame(purchases, columns=columns))

    return df.reset_index(drop=True)  # Reset index for the resulting DataFrame


def pareto_nbd_model(T, r, alpha, s, beta, size=1):
    """
    Generate artificial data according to the population-level distribution class
    for a continuous, non-contractual, Pareto/NBD process.

    It is based on Schmittlein, et al. in [2].

    The Pareto/NBD model is similar to BG/NBD but introduces a different model
    for customer death, using a gamma distribution for the customer lifetime.

    See [2]_ for model details.

    Parameters
    ----------
    T: array_like
        The length of time observing new customers.
    r, alpha, s, beta: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate. Default is 1.

    Returns
    -------
    pandas.DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'mu', 'alive', 'customer_id'

    References
    ----------
    .. [2]: David C. Schmittlein, Donald G. Morrison and Richard Colombo.
        "Counting Your Customers: Who Are They and What Will They Do Next."
        Management Science, Vol. 33, No. 1 (Jan., 1987), pp. 1-24.
    .. [3]: Fader, Peter & G. S. Hardie, Bruce (2005).
        "A Note on Deriving the Pareto/NBD Model and Related Expressions."
        http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf

    """
    # Handle the case where T is a scalar or an array
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    # Generate random purchase rates (Gamma distribution)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    # Generate random customer lifetimes (Gamma distribution)
    mus = np.random.gamma(s, scale=1.0 / beta, size=size)

    # Initialize columns for the resulting DataFrame
    columns = ["frequency", "recency", "T", "lambda", "mu", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    # Loop over each customer to simulate their purchase behavior
    for i in range(size):
        l = lambda_[i]
        mu = mus[i]
        time_of_death = np.random.exponential(scale=1.0 / mu)

        # hacky until I can find something better
        # Simulate times between purchases (exponentially distributed) until death or observation ends
        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        while np.sum(times) + next_purchase_in < min(time_of_death, T[i]):
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)

        times = np.array(times).cumsum()
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[0],
            np.max(times if times.shape[0] > 0 else 0),
            T[i],
            l,
            mu,
            time_of_death
            > T[i],  # Whether the customer survives the observation period
            i,
        )

    return df.set_index(
        "customer_id"
    )  # Return the DataFrame with customer IDs as the index


def modified_beta_geometric_nbd_model(T, r, alpha, a, b, size=1):
    """
    Generate artificial data according to the MBG/NBD model.

    See [3]_, [4]_ for model details

    Parameters
    ----------
    T: array_like
        The length of time observing new customers.
    r, alpha, a, b: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'T', 'lambda', 'p', 'alive', 'customer_id'

    References
    ----------
    .. [1]: '"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model'
       (http://brucehardie.com/papers/bgnbd_2004-04-20.pdf)
    .. [2] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.

    """
    if type(T) in [float, int]:
        T = T * np.ones(size)
    else:
        T = np.asarray(T)

    probability_of_post_purchase_death = np.random.beta(a, b, size=size)
    lambda_ = np.random.gamma(r, scale=1.0 / alpha, size=size)

    columns = ["frequency", "recency", "T", "lambda", "p", "alive", "customer_id"]
    df = pd.DataFrame(np.zeros((size, len(columns))), columns=columns)

    for i in range(size):
        p = probability_of_post_purchase_death[i]
        l = lambda_[i]

        # hacky until I can find something better
        times = []
        next_purchase_in = np.random.exponential(scale=1.0 / l)
        alive = (
            np.random.np.random() > p
        )  # essentially the difference between this model and BG/NBD
        while (np.sum(times) + next_purchase_in < T[i]) and alive:
            times.append(next_purchase_in)
            next_purchase_in = np.random.exponential(scale=1.0 / l)
            alive = np.random.np.random() > p

        times = np.array(times).cumsum()
        df.iloc[i] = (
            np.unique(np.array(times).astype(int)).shape[0],
            np.max(times if times.shape[0] > 0 else 0),
            T[i],
            l,
            p,
            alive,
            i,
        )

    return df.set_index("customer_id")


def beta_geometric_beta_binom_model(N, alpha, beta, gamma, delta, size=1):
    """
    Generate artificial data according to the population-level distribution class
    for a discrete, non-contractual, Beta-Geometric/Beta-Binomial process.

    You may wonder why we can have frequency = n_periods, when frequency excludes their
    first order. When a customer purchases something, they are born, _and in the next
    period_ we start asking questions about their alive-ness. So really they customer has
    bought frequency + 1, and been observed for n_periods + 1

    Parameters
    ----------
    N: array_like
        Number of transaction opportunities for new customers.
    alpha, beta, gamma, delta: float
        Parameters in the model. See [1]_
    size: int, optional
        The number of customers to generate

    Returns
    -------
    DataFrame
        with index as customer_ids and the following columns:
        'frequency', 'recency', 'n_periods', 'lambda', 'p', 'alive', 'customer_id'

    References
    ----------
    .. [1]: Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
        "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
        Marketing Science, 29 (6), 1086-1108.
        https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf

    """
    if type(N) in [float, int, np.int64]:
        N = N * np.ones(size)
    else:
        N = np.asarray(N)

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

        # hacky until I can find something better
        current_t = 0
        alive = True
        times = []
        while current_t < N[i] and alive:
            alive = np.random.binomial(1, theta) == 0
            if alive and np.random.binomial(1, p) == 1:
                times.append(current_t)
            current_t += 1
        # adding in final death opportunity to agree with [1]
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
