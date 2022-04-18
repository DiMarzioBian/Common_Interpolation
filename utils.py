import numpy as np
from numpy.linalg import solve


def get_ctrl_pts(args, x, y):
    """ Calculate control points. """
    t = [0, *np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2).cumsum().tolist()]
    t = [x / max(t) for x in t]

    # determine D vector for bot x and y
    vec_D_x = np.concatenate(([x[0]], [0], x[1:-1], [0], [x[-1]]))
    vec_D_y = np.concatenate(([y[0]], [0], y[1:-1], [0], [y[-1]]))

    # determine matrix N
    n = len(x) - 1
    mat_N = np.zeros((len(x) + 2, len(x) + 2))
    knots = np.array([0, 0, 0] + t + [1, 1, 1])

    # matrix N center part
    for i in range(len(x)):
        for j in range(len(x) + 2):
            if j + 3 < len(knots) and knots[j] <= t[i] < knots[j + 1]:

                mat_N[i + 1][j] = ((t[i] - knots[j]) ** 3) / (
                        (knots[j + 1] - knots[j]) * (knots[j + 2] - knots[j]) * (knots[j + 3] - knots[j]))

            elif j + 4 < len(knots) and knots[j + 1] <= t[i] < knots[j + 2]:

                mat_N[i + 1][j] += ((t[i] - knots[j]) ** 2 * (knots[j + 2] - t[i])) / (
                        (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j]) * (knots[j + 2] - knots[j]))

                mat_N[i + 1][j] += ((knots[j + 3] - t[i]) * (t[i] - knots[j]) * (t[i] - knots[j + 1])) / (
                        (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j + 1]) * (knots[j + 3] - knots[j]))

                mat_N[i + 1][j] += ((knots[j + 4] - t[i]) * (t[i] - knots[j + 1]) ** 2) / (
                        (knots[j + 2] - knots[j + 1]) * (knots[j + 4] - knots[j + 1]) * (knots[j + 3] - knots[j + 1]))

            elif j + 4 < len(knots) and knots[j + 2] <= t[i] < knots[j + 3]:

                mat_N[i + 1][j] += ((t[i] - knots[j]) * (knots[j + 3] - t[i]) ** 2) / (
                        (knots[j + 3] - knots[j + 2]) * (knots[j + 3] - knots[j + 1]) * (knots[j + 3] - knots[j]))

                mat_N[i + 1][j] += ((knots[j + 4] - t[i]) * (knots[j + 3] - t[i]) * (t[i] - knots[j + 1])) / (
                        (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]) * (knots[j + 3] - knots[j + 1]))

                mat_N[i + 1][j] += ((knots[j + 4] - t[i]) ** 2 * (t[i] - knots[j + 2])) / (
                        (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]))

            elif j + 4 < len(knots) and knots[j + 3] <= t[i] < knots[j + 4]:
                mat_N[i + 1][j] += ((knots[j + 4] - t[i]) ** 3) / ((knots[j + 4] - knots[j + 3]) *
                                                                   (knots[j + 4] - knots[j + 2]) *
                                                                   (knots[j + 4] - knots[j + 1]))

    # matrix N endpoint conditions: top left
    mat_N[0][0] = 1
    mat_N[1][0] = 6 * (knots[4] - t[0]) / ((knots[4] - knots[3]) * (knots[4] - knots[2]) * (knots[4] - knots[1]))

    mat_N[1][1] += (2 * (t[0] - knots[1]) - 4 * (knots[4] - t[0])) / (
            (knots[4] - knots[3]) * (knots[4] - knots[2]) * (knots[4] - knots[1]))
    mat_N[1][1] += (-2 * (knots[5] - t[0]) - 2 * (knots[4] - t[0]) + 2 * (t[0] - knots[2])) / (
            (knots[4] - knots[3]) * (knots[5] - knots[2]) * (knots[4] - knots[2]))
    mat_N[1][1] += (4 * (knots[5] - t[0]) - 2 * (t[0] - knots[3])) / (
            (knots[4] - knots[3]) * (knots[5] - knots[3]) * (knots[5] - knots[2]))

    mat_N[1][2] += (-2 * (knots[4] - t[0])) / (
            (knots[4] - knots[3]) * (knots[5] - knots[2]) * (knots[4] - knots[2]))
    mat_N[1][2] += (2 * (knots[5] - t[0]) - 2 * (t[0] - knots[2]) - 2 * (t[0] - knots[3])) / (
            (knots[4] - knots[3]) * (knots[5] - knots[3]) * (knots[5] - knots[2]))
    mat_N[1][2] += (2 * (knots[6] - t[0]) - 4 * (t[0] - knots[3])) / (
            (knots[4] - knots[3]) * (knots[6] - knots[3]) * (knots[5] - knots[3]))

    # matrix N endpoint conditions: bottom right
    mat_N[-1][-1] = 1
    mat_N[-2][-1] = (6 * (t[n] - knots[n + 2])) / (
                (knots[n + 3] - knots[n + 2]) * (knots[n + 3] - knots[n + 2]) * (knots[n + 5] - knots[n + 2]))

    mat_N[-2][-2] += (-2 * (knots[n + 3] - t[-1])) / (
            (knots[n + 4] - knots[n + 2]) * (knots[n + 4] - knots[n + 1]) * (knots[n + 3] - knots[n + 1]))
    mat_N[-2][-2] += (2 * (knots[n + 4] - t[-1]) - 2 * (t[-1] - knots[n + 1]) - 2 * (t[-1] - knots[n + 2])) / (
            (knots[n + 3] - knots[n + 2]) * (knots[n + 4] - knots[n + 2]) * (knots[n + 4] - knots[n + 1]))
    mat_N[-2][-2] += (2 * (knots[n + 5] - t[-1]) - 4 * (t[-1] - knots[n + 2])) / (
            (knots[n + 3] - knots[n + 2]) * (knots[n + 5] - knots[n + 2]) * (knots[n + 5] - knots[n + 2]))

    mat_N[-2][-3] += (2 * (t[-1] - knots[n]) - 4 * (knots[n + 3] - t[-1])) / (
            (knots[n + 3] - knots[n + 2]) * (knots[n + 3] - knots[n + 1]) * (knots[n + 3] - knots[n]))
    mat_N[-2][-3] += (-2 * (knots[n + 4] - t[-1]) - 2 * (knots[n + 3] - t[-1]) + 2 * (t[-1] - knots[n + 1])) / (
            (knots[n + 3] - knots[n + 2]) * (knots[n + 4] - knots[n + 1]) * (knots[n + 3] - knots[n + 1]))
    mat_N[-2][-3] += (4 * (knots[n + 4] - t[-1]) - 2 * (t[-1] - knots[n + 2])) / (
            (knots[n + 4] - knots[n + 2]) * (knots[n + 4] - knots[n + 2]) * (knots[n + 4] - knots[n + 1]))

    x_ctrl_pts = solve(mat_N, vec_D_x)
    y_ctrl_pts = solve(mat_N, vec_D_y)

    print('\n[info] Printing linear system:')
    print('\n\tx_ctrl_pts:\n\t\t{x_ctrl_pts}'.format(x_ctrl_pts=x_ctrl_pts))
    print('\n\ty_ctrl_pts:\n\t\t{y_ctrl_pts}'.format(y_ctrl_pts=y_ctrl_pts))
    print('\n\tmat_N:\n{mat_N}'.format(mat_N=mat_N))

    return x_ctrl_pts, y_ctrl_pts, knots
