from sklearn.linear_model import LinearRegression


def linear_model(X, y):
    """
    Fit a multiple linear regression model to X and y

    :param X: nxm np.array of features
    :param y: nx1 np.array of values
    :return: sklearn model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model
