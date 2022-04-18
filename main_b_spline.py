import argparse
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from utils import get_ctrl_pts

h = 3


def main():
    parser = argparse.ArgumentParser(description='Interpolate using cubic B-spline interpolation algorithm')

    # parameter settings
    parser.add_argument('--sampling', type=str, default='uniform', help='sampling strategy',
                        choices=['uniform', 'chord_ux', 'chord_uy', 'chord_xy', 'chord_uxy'])
    parser.add_argument('--degree', type=int, default=3, help='B-spline degree')
    parser.add_argument('--n_curve', type=int, default=100, help='number of point to plot curve')
    parser.add_argument('--n_sample', type=int, default=30, help='number of point before interpolation')

    # path settings
    parser.add_argument('--path_img', type=str, default='./images/', help='path of images')
    args = parser.parse_args()

    if args.sampling == 'uniform':
        u_sample = np.linspace(0, 1, args.n_sample, endpoint=False)

    elif args.sampling == 'chord_ux':
        u_sample_uni = np.linspace(0, 1, args.n_sample, endpoint=False)
        x_sample_uni = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.cos(12.2 * u_sample_uni)

        u_sample = [0,
                    *np.sqrt((u_sample_uni[1:] - u_sample_uni[:-1]) ** 2 + (x_sample_uni[1:] - x_sample_uni[:-1]) ** 2
                             ).cumsum().tolist()]
        u_sample = np.array([x / max(u_sample) for x in u_sample])

    elif args.sampling == 'chord_uy':
        u_sample_uni = np.linspace(0, 1, args.n_sample, endpoint=False)
        y_sample_uni = (np.exp(np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.sin(12.2 * u_sample_uni)

        u_sample = [0,
                    *np.sqrt((u_sample_uni[1:] - u_sample_uni[:-1]) ** 2 + (y_sample_uni[1:] - y_sample_uni[:-1]) ** 2
                             ).cumsum().tolist()]
        u_sample = np.array([x / max(u_sample) for x in u_sample])

    elif args.sampling == 'chord_xy':
        u_sample_uni = np.linspace(0, 1, args.n_sample, endpoint=False)
        x_sample_uni = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.cos(12.2 * u_sample_uni)
        y_sample_uni = (np.exp(np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.sin(12.2 * u_sample_uni)

        u_sample = [0,
                    *np.sqrt((x_sample_uni[1:] - x_sample_uni[:-1]) ** 2 + (y_sample_uni[1:] - y_sample_uni[:-1]) ** 2
                             ).cumsum().tolist()]
        u_sample = np.array([x / max(u_sample) for x in u_sample])

    elif args.sampling == 'chord_uxy':
        u_sample_uni = np.linspace(0, 1, args.n_sample, endpoint=False)
        x_sample_uni = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.cos(12.2 * u_sample_uni)
        y_sample_uni = (np.exp(np.sin(6.2 * u_sample_uni - 0.027 * h)) + 0.1) * np.sin(12.2 * u_sample_uni)

        u_sample = [0,
                    *np.sqrt((u_sample_uni[1:] - u_sample_uni[:-1]) ** 2 + (x_sample_uni[1:] - x_sample_uni[:-1]) ** 2
                             + (y_sample_uni[1:] - y_sample_uni[:-1]) ** 2).cumsum().tolist()]
        u_sample = np.array([x / max(u_sample) for x in u_sample])

    else:
        raise ValueError('Chosen sampling strategy: {sampling} not implemented.'.format(sampling=args.sampling))

    # original curve
    u_curve = np.linspace(0, 1, args.n_curve, endpoint=True)
    x_curve = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_curve - 0.027 * h)) + 0.1) * np.cos(12.2 * u_curve)
    y_curve = (np.exp(np.sin(6.2 * u_curve - 0.027 * h)) + 0.1) * np.sin(12.2 * u_curve)

    # sampling
    x_sample = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_sample - 0.027 * h)) + 0.1) * np.cos(12.2 * u_sample)
    y_sample = (np.exp(np.sin(6.2 * u_sample - 0.027 * h)) + 0.1) * np.sin(12.2 * u_sample)

    # fitting
    x_ctrl_pts, y_ctrl_pts, knots = get_ctrl_pts(args, x_sample, y_sample)
    tck = [knots, [x_ctrl_pts, y_ctrl_pts], args.degree]
    t = np.linspace(0, 1, args.n_sample, endpoint=True)
    (x_fit, y_fit) = interpolate.splev(t, tck)

    # plotting
    plt.figure()
    plt.plot(x_curve, y_curve, 'k--', label='Original curve', marker='.')
    plt.plot(x_sample, y_sample, 'b--', label='Sampled curve', marker='.')
    plt.plot(x_fit, y_fit, label='Fit curve', marker='.', color='red')
    plt.legend(loc='best')
    plt.title('Fitting curve r(u) with sampling: {sampling} on {n_sample} points'
              .format(sampling=args.sampling, n_sample=args.n_sample))
    plt.savefig(args.path_img + 'cubic_b_spline_' +args.sampling + '_' + str(args.n_sample) +
                '.png')
    plt.show()


if __name__ == '__main__':
    main()
