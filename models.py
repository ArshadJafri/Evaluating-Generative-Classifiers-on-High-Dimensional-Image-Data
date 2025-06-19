import numpy as np

class LDAModel:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # Calculate the class means
        self.means = np.array([X[y == cls].mean(axis=0) for cls in self.classes])
        
        # Calculate the overall mean
        overall_mean = X.mean(axis=0)

        # Calculate the within-class scatter matrix SW and between-class scatter matrix SB
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for i, cls in enumerate(self.classes):
            X_class = X[y == cls]
            class_scatter = np.dot((X_class - self.means[i]).T, (X_class - self.means[i]))
            SW += class_scatter

            n_class_samples = X_class.shape[0]
            mean_diff = (self.means[i] - overall_mean).reshape(n_features, 1)
            SB += n_class_samples * np.dot(mean_diff, mean_diff.T)

        # Calculate the weight matrix using the eigenvalue problem (SB, SW)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
        self.w = self.eigenvectors[:, :len(self.classes) - 1]  # Keep only the top components

    def predict(self, X):
        X_projected = np.dot(X, self.w)
        
        dists = [np.linalg.norm(X_projected - np.dot(self.means[i], self.w), axis=1) for i in range(len(self.classes))]
        return self.classes[np.argmin(dists, axis=0)]


class QDAModel:
    def __init__(self, reg_factor=1e-5):
        self.reg_factor = reg_factor
        self.class_means = None
        self.cov_matrices = None
        self.priors = None

    def fit(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = len(classes)
        self.class_means = np.zeros((n_classes, n_features))
        self.cov_matrices = []
        self.priors = np.zeros(n_classes)


        for i, cls in enumerate(classes):
            X_cls = X[y == cls]
            self.class_means[i, :] = X_cls.mean(axis=0)
            cov_matrix = np.cov(X_cls, rowvar=False) 
            cov_matrix += self.reg_factor * np.eye(n_features)

            self.cov_matrices.append(cov_matrix)
            self.priors[i] = X_cls.shape[0] / X.shape[0]

    def predict(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.class_means)
        log_likelihoods = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            mean = self.class_means[i]
            cov_matrix = self.cov_matrices[i]
            cov_inv = np.linalg.inv(cov_matrix)  # Inverse of covariance matrix

            # Avoid overflow by using log-determinant
            _, log_cov_det = np.linalg.slogdet(cov_matrix)

            # Calculate the log-likelihood using the quadratic form
            diff = X - mean
            log_likelihood = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)
            log_likelihood -= 0.5 * log_cov_det
            log_likelihood -= 0.5 * n_features * np.log(2 * np.pi)
            log_likelihoods[:, i] = log_likelihood + np.log(self.priors[i])

        # Predict the class with the highest log likelihood
        return np.argmax(log_likelihoods, axis=1)
    
class GaussianNBModel:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.class_priors = {}

        for cls in self.classes:
            X_class = X[y == cls]
            self.means[cls] = X_class.mean(axis=0)
            self.variances[cls] = X_class.var(axis=0)
            self.class_priors[cls] = X_class.shape[0] / X.shape[0]

    def predict(self, X):
        posteriors = []

        for cls in self.classes:
            mean = self.means[cls]
            var = self.variances[cls]
            prior = np.log(self.class_priors[cls])
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
            log_posterior = log_likelihood + prior
            posteriors.append(log_posterior)

        return self.classes[np.argmax(posteriors, axis=0)]
