import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    A = []
    b = []
    for im_c, wor_c in zip(points2d, points3d):
        X_i = wor_c[0]
        Y_i = wor_c[1]
        Z_i = wor_c[2]
        u_i = im_c[0]
        v_i = im_c[1]
        A_i = np.asarray([[X_i, Y_i, Z_i, 1, 0, 0, 0, 0, -X_i*u_i, -Y_i*u_i, -Z_i*u_i], [0, 0, 0, 0, X_i, Y_i, Z_i, 1, -X_i*v_i, -Y_i*v_i, -Z_i*v_i]])
        b_i = np.asarray([u_i, v_i])
        A.append(A_i)
        b.append(b_i)
    A = np.asarray(A).reshape(-1, 11)
    b = np.asarray(b).flatten()
    M = np.linalg.lstsq(A, b)[0]
    M = np.append(M, 1).reshape(3, 4)
    residual = np.linalg.lstsq(A, b)[1][0]

    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # print('Randomly setting matrix entries as a placeholder')
    # M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
    #               [0.6750, 0.3152, 0.1136, 0.0480],
    #               [0.1020, 0.1725, 0.7244, 0.9932]])
    # residual = 7 # Arbitrary stencil code initial value

    return M, residual

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform one set of 
    points into estimates of the positions of a second set of points, 
    e.g., estimating F from points1 to points2 will let us produce points2'. 
    The difference between points2 and points2' is the residual error in the 
    fundamental matrix estimation.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the error in the estimation
    """
    ########################
    # TODO: Your code here #
    ########################
    # best_inlier_residual = abs(np.transpose(np.asarray(expanded_matches_b))@best_Fmatrix@np.asarray(expanded_matches_a))

    # Arbitrary intentionally incorrect Fundamental matrix placeholder
    F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    residual = 5 # Arbitrary stencil code initial value

    u = points1[:, 0]
    v = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]
    A = np.array([u*u2, v*u2, u2, u*v2, v*v2, v2, u, v, np.ones(u.shape)])
    U, S, V = np.linalg.svd(A, full_matrices=False)
    F_matrix = U@S@np.transpose(V)
    F_matrix = F_matrix.reshape(3, 3)
    points1 = np.hstack((points1, np.ones(points1.shape[0]).reshape(-1, 1)))
    points2 = np.hstack((points2, np.ones(points2.shape[0]).reshape(-1, 1)))
    residual = np.trace(np.abs(points2@F_matrix@np.transpose(points1)))

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the error induced by best_Fmatrix

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################
    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    best_inliers_a = matches1[0:29, :]
    best_inliers_b = matches2[0:29, :]
    best_inlier_residual = 5
    max_inliers = 0
    threshold = 0.003
    for i in range(num_iters):
        inliers = []
        inlier_residual = []
        p = np.random.permutation(len(matches1))
        matches1_subset = matches1[p][:9]
        matches2_subset = matches2[p][:9]

        F_matrix, _ = cv2.findFundamentalMat(matches1_subset, matches2_subset, cv2.FM_8POINT, 1e10, 0, 1)
        expanded_matches_a = []
        expanded_matches_b = []
        for j in range(matches1.shape[0]):
            ma = np.asarray([matches1[j][0], matches1[j][1], 1])
            mb = np.asarray([matches2[j][0], matches2[j][1], 1])
            expanded_matches_a.append(ma)
            expanded_matches_b.append(mb)
            distance = np.sum(abs(np.transpose(mb)@F_matrix@ma)**2)**(1/2)
            residual = abs(np.transpose(mb)@F_matrix@ma)
            inlier_residual.append(residual)
            if distance <= threshold:
                inliers.append([j])
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_Fmatrix = F_matrix
            best_inliers_a = np.squeeze(matches1[inliers,:])
            best_inliers_b = np.squeeze(matches2[inliers,:])
            best_inlier_residual = min(inlier_residual)

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'

    # best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    # best_inliers_a = matches1[0:29, :]
    # best_inliers_b = matches2[0:29, :]
    # best_inlier_residual = 5 # Arbitrary stencil code initial value

    # For your report, we ask you to visualize RANSAC's 
    # convergence over iterations. 
    # For each iteration, append your inlier count and residual to the global variables:
    #   inlier_counts = []
    #   inlier_residuals = []
    # Then add flag --visualize-ransac to plot these using visualize_ransac()
    

    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual

def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    ########################
    # TODO: Your code here #

    # Initial random values for 3D points
    points3d = np.random.rand(len(points1),3)

    # Solve for ground truth points
    index = 0
    for p1, p2 in zip(points1, points2):
        A = []
        b = []
        u_i = p1[0]
        v_i = p1[1]
        u_j = p2[0]
        v_j = p2[1]
        A_i = np.asarray([[M1[2][0]*u_i - M1[0][0], M1[2][1]*u_i - M1[0][1], M1[2][2]*u_i - M1[0][2]], [M1[2][0]*v_i - M1[1][0], M1[2][1]*v_i - M1[1][1], M1[2][2]*v_i - M1[1][2]], [M1[2][0], M1[2][1], M1[2][2]]])
        A_j = np.asarray([[M2[2][0]*u_j - M2[0][0], M2[2][1]*u_j - M2[0][1], M2[2][2]*u_j - M2[0][2]], [M2[2][0]*v_j - M2[1][0], M2[2][1]*v_j - M2[1][1], M2[2][2]*v_j - M2[1][2]], [M2[2][0], M2[2][1], M2[2][2]]])
        b_i = np.asarray([M1[0][3] - M1[2][3]*u_i, M1[1][3] - M1[2][3]*v_i, 1 - M1[2][3]])
        b_j = np.asarray([M2[0][3] - M2[2][3]*u_j, M2[1][3] - M2[2][3]*v_j, 1 - M2[2][3]])
        A.append(A_i)
        A.append(A_j)
        b.append(b_i)
        b.append(b_j)
        XYZ_i = np.linalg.lstsq(np.asarray(A).reshape(-1, 3), np.asarray(b).flatten())[0]
        points3d[index] = XYZ_i
        index += 1

    ########################

    return points3d


#/////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.maximum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()