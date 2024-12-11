from segmentation.image_segmenter import ImageSegmenter
import numpy as np
from scipy.signal import convolve2d

__min_clip = 1e-5

def get_d(arr):
    # Check if the array has 3 dimensions or 2 dimensions
    if arr.ndim == 2:
        # If the array has 2 dimensions, assume d = 1
        return 1
    elif arr.ndim == 3:
        # If the array has 3 dimensions, return d (the third dimension size)
        return arr.shape[2]
    else:
        raise ValueError("Array must have 2 or 3 dimensions")

class FuzzyCMeansSegmenter(ImageSegmenter):

    def __init__(self, n_groups: int, target_error: float, neighbourhood_size: int, m: float, lambda_val: float,
                 max_iter: int):
        # TODO finish arg parsing + default values
        
        self.n_groups = n_groups
        self.target_error = target_error
        self.neighbourhood_size = neighbourhood_size
        self.m = m
        self.lambda_val = lambda_val
        self.max_iter = max_iter

        self.p = 1
        self.q = 1
        self.epsilon = 0

    def segment_image(self, image: np.ndarray, previous_guess: np.ndarray = None) -> np.ndarray:

        # TODO use previous_guess

        kernel = np.ones((self.neighbourhood_size,) * 2)

        # Update centroids first
        default_u = np.zeros([self.n_groups, ] + list(image.shape[0:2]))
        rows, cols = image.shape[0:2]
        row_indices, col_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        summed = row_indices * cols + col_indices

        for ii in range(self.n_groups):
            idxii = summed % self.n_groups == ii
            default_u[ii, idxii] = 1

        centroids = self.__compute_centroids(image, default_u)

        # Warning dismiss
        u_prime_prime = None

        for i in range(self.max_iter):

            # Compute u
            u = self.__update_u(image, centroids)

            # Compute hesitation
            ones = np.ones_like(u)
            pi = ones - u - (ones - u) / (ones + self.lambda_val * u)

            # Compute membership function
            u_prime = u + pi

            # Compute spatial function
            h = np.zeros_like(u)
            for j in range(u.shape[0]):
                h[j, :, :] = convolve2d(u[j, :, :], kernel, mode='same', boundary='fill')

            # Compute new membership function
            u_prime_prime_prod = np.pow(u_prime, self.p) * np.pow(h, self.q)
            u_prime_prime = u_prime_prime_prod / np.sum(u_prime_prime_prod, axis=0)[np.newaxis, :, :]

            # Compute centroids
            centroids = self.__compute_centroids(image, u_prime_prime)

            # TODO check u(new) - u(old) diff

            if self.__compute_error(image, centroids, u_prime_prime) < self.target_error:
                break

        return np.argmax(u_prime_prime, axis = 0)

    def __update_u(self, image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        '''
            Computes the membership value according to paper specification
        '''
        dists = self.__compute_pixel_dist(image, centroids)
        summed = np.sum(dists, axis=0)[np.newaxis, :, :]

        # clip to avoid runtime issues (nan/inf)
        dists = np.clip(dists, a_min = __min_clip, a_max = np.inf)

        denom = np.pow(dists / summed, 2.0 / (self.m - 1))

        u_ones = np.ones_like(denom)

        return u_ones / denom

    def __compute_centroids(self, image: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Computes the new centroid position according to paper specification
        '''
        numerator = np.sum(np.pow(u, self.m)[:, :, :, np.newaxis] * image[np.newaxis, :, :], axis=(1, 2))
        denominator = np.sum(np.pow(u, self.m), axis=(1, 2))

        return numerator / denominator[:, np.newaxis]

    def __compute_error(self, image: np.ndarray, centroids: np.ndarray, classification: np.ndarray) -> float:
        '''
            Computes the error
        '''
        # TODO assert centroids length = third dim classification
        error = self.__compute_pixel_dist(image, centroids)
        expanded_classification = np.pow(classification[:, np.newaxis, np.newaxis], self.m)

        return np.sum(error * expanded_classification)

    def __compute_pixel_dist(self, pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        '''
            Return L2 norm squared of between pixels and centroids

            pixels: image (N, M, D) to compute the individual distances (np.ndarray)
            centroids: centroids (C, D) to compute distance from (np.ndarray)

            returns float pixel distance array (C, N, M, D)
        '''
        expanded_image = np.array(pixels[np.newaxis, :, :, :], dtype=float)
        expanded_centroids = np.array(centroids[:, np.newaxis, np.newaxis, :], dtype=float)

        diff = expanded_image - expanded_centroids
        pow_mat = np.pow(diff, self.m)

        return np.sum(pow_mat, axis=3)