"""
This model is based off of HW5.

Specifically, we use Non-negative Matrix Factorization.
"""
import numpy as np
from utils import read_data, K
from tqdm import tqdm


def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """

    return (reg * Ui) - Vj * (Yij - np.dot(Ui, Vj))


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return (reg * Vj) - Ui * (Yij - np.dot(Ui, Vj))


def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    U_norm = np.linalg.norm(U)
    V_norm = np.linalg.norm(V)

    U_norm_squared = U_norm ** 2
    V_norm_squared = V_norm ** 2

    reg_term = (reg / 2) * (U_norm_squared + V_norm_squared)

    total_error_term = 0

    N = Y.shape[0]
    for index in range(N):
        i, j, Y_ij = Y[index]

        i -= 1
        j -= 1

        predictions = np.dot(U[i], V[:, j])
        difference = Y_ij - predictions

        squared_difference = np.square(difference)
        total_error_term += squared_difference

    avg_error = ((1/2) * total_error_term) / N
    return avg_error


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.uniform(-0.5, 0.5, size=(M, K))
    V = np.random.uniform(-0.5, 0.5, size=(K, N))

    train_data_size = Y.shape[0]

    first_two_losses = []
    last_two_losses = []

    first_two_losses.append(get_err(U, V, Y, reg))

    for epoch in tqdm(range(max_epochs)):
        if len(first_two_losses) == 2 and len(last_two_losses) == 2:
            delta_1 = first_two_losses[1] - first_two_losses[0]
            delta_t = last_two_losses[1] - last_two_losses[0]

            if delta_t / delta_1 <= eps:
                break

        indices = np.random.permutation(train_data_size)
        shuffled_train_data = Y[indices]

        for index in indices:
            i, j, Y_ij = shuffled_train_data[index]

            i -= 1
            j -= 1

            U[i] -= eta * grad_U(U[i], Y_ij, V[:, j], reg, eta)
            V[:, j] -= eta * grad_V(V[:, j], Y_ij, U[i], reg, eta)

        cur_loss = get_err(U, V, Y, reg)

        # TODO: handle edge case for `last_two_losses`
        if epoch == 0:
            first_two_losses.append(cur_loss)
        elif epoch == 1:
            last_two_losses.append(cur_loss)
        else:
            last_two_losses = [last_two_losses[-1], cur_loss]

    error = get_err(U, V, Y)  # note: reg = 0.0

    return U, V, error


def compute_num_users_and_movies(data):
    num_users = np.max(data['User ID']) + 1  # User IDs are zero-indexed
    num_movies = np.max(data['Movie ID']) + 1  # Movie IDs are zero-indexed

    return num_users, num_movies


def get_U_V(data):
    num_users, num_movies = compute_num_users_and_movies(data)
    eta = 0.03
    reg = 0.1
    eps = 0.0001
    max_epochs = 300

    data_as_numpy = data.to_numpy(copy=True).astype(int)

    U_transpose, V, _ = train_model(num_users, num_movies, K, eta, reg,
                                    data_as_numpy, eps=eps, max_epochs=max_epochs)

    # Desired output shapes:
    # U: (K, M)
    # V: (K, N)

    U = U_transpose.T

    return U, V
