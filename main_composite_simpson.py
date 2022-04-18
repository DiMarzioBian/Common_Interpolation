import argparse
import numpy as np

h = 3


def main():
    parser = argparse.ArgumentParser(description='Compute integral using composite Simpson’s rule')

    # parameter settings
    parser.add_argument('--n_sample', type=int, default=1001, help='samples to plot')
    args = parser.parse_args()

    # calculating arrays
    u = np.linspace(0, 1, args.n_sample, endpoint=True)
    x = 1.5 * (np.exp(1.5 * np.sin(6.2 * u - 0.027 * h)) + 0.1) * np.cos(12.2 * u)

    # calculating integral
    N = args.n_sample // 2
    height = 1. / (args.n_sample - 1)
    integral_x = x[0] + x[-1]
    integral_x += 4 * sum([x[2 * i - 1] for i in range(1, N + 1)]) + 2 * sum([x[2 * i] for i in range(1, N)])
    integral_x *= height / 3

    print('\n[info] Integral of x(u) is {integral}.\n'.format(integral=integral_x))


if __name__ == '__main__':
    main()
