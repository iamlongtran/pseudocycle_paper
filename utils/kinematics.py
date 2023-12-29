# Module for basic manipulations of atoms/crds/angles/proteins in numpy and torch 
import numpy as np 
import torch 

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    credit to: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def np_kabsch(A,B):
    """
    Numpy version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) np.array - shape (N,3) arrays of xyz crds of points


    Returns:
        rms - rmsd between A and B
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """
    A = np.copy(A)
    B = np.copy(B)

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    def rmsd(V,W, eps=0):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)


    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U,S,Vt = np.linalg.svd(C)


    # ensure right handed coordinate system
    d = np.eye(3)
    d[-1,-1] = np.sign(np.linalg.det(Vt.T@U.T))

    # construct rotation matrix
    R = Vt.T@d@U.T

    # get rotated coords
    rB = B@R

    # calculate rmsd
    rms = rmsd(A,rB)

    return rms, rB, R


def th_kabsch(A,B):
    """
    Torch version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) torch tensor - shape (N,3) arrays of xyz crds of points


    Returns:
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return torch.mean(X, dim=0, keepdim=True)

    def rmsd(V,W, eps=1e-6):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(-2,-1)) / N + eps)


    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U,S,Vt = torch.svd(C)

    # ensure right handed coordinate system
    d = torch.eye(3)
    d[-1,-1] = torch.sign(torch.det(Vt@U.T))

    # construct rotation matrix
    R = Vt@d@U.T

    # get rotated coords
    rB = B@R

    # calculate rmsd
    rms = rmsd(A,rB)

    return rms, rB, R

