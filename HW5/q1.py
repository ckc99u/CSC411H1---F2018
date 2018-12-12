"""
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
"""

import data
import numpy as np
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    # Compute means
    _, D = train_data.shape
    K = len(np.unique(train_labels))

    means = np.zeros((K, D))

    for i in range(K):
        indices = train_data[train_labels == i]
        means[i] = np.mean(indices, axis=0)

    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    _, D = train_data.shape
    K = len(np.unique(train_labels))

    covariances = np.zeros((K, D, D))
    # Compute covariances

    for i in range(K):
        indices = train_data[train_labels == i]
        indices -= np.mean(indices, axis=0)
        covariances[i] = np.dot(np.transpose(indices), indices) / float(indices.shape[0]) + 0.01 * np.eye(D)

    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """
    N, D = digits.shape
    K = covariances.shape[0]

    gen_log_lik = np.zeros((N, K))

    for i in range(K):
        x_mu_k = digits - means[i]

        gen_log_lik[:, i] = -0.5 * (D * np.log(2 * np.pi) +
                                    np.log(np.linalg.det(covariances[i])) +
                                    np.diag(np.dot(np.dot(x_mu_k, np.linalg.inv(covariances[i])), x_mu_k.T)))

    return gen_log_lik


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """
    N, _ = digits.shape
    K = covariances.shape[0]

    cond_log_lik = np.zeros((N, K))
    gen_log_lik = generative_likelihood(digits, means, covariances)

    P_y = 0.1
    P_x = np.sum(np.exp(gen_log_lik) * P_y, axis=1)

    for i in range(K):
        cond_log_lik[:, i] = gen_log_lik[:, i] + np.log(P_y) - np.log(P_x)

    return cond_log_lik


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    K = len(np.unique(labels))

    avg_cond_lik = 0

    for i in range(K):
        cur_cond_lik = cond_likelihood[labels == i]
        avg_cond_lik += np.sum(cur_cond_lik, axis=0)[i]

    return avg_cond_lik * digits.shape[0] ** -1


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class

    return np.argmax(cond_likelihood, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('hw5digits')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # (a)
    print avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print avg_conditional_likelihood(test_data, test_labels, means, covariances)

    # (b)
    most_lik_class_train = classify_data(train_data, means, covariances)
    most_lik_class_test = classify_data(test_data, means, covariances)

    print 100 * float(np.count_nonzero(train_labels == most_lik_class_train)) / len(train_labels)
    print 100 * float(np.count_nonzero(test_labels == most_lik_class_test)) / len(test_labels)

    # (c)
    for i in range(len(np.unique(train_labels))):
        class_cov_matr = covariances[i]
        eig_vals, eig_vecs = np.linalg.eig(class_cov_matr)

        leading_eig_vec = eig_vecs[np.argmax(eig_vals)]

        plt.show()

        plt.imshow(np.reshape(leading_eig_vec, (8, 8)), cmap="Greys")


if __name__ == '__main__':
    main()
