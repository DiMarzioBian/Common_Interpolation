import argparse
import numpy as np
import matplotlib.pyplot as plt

h = 3


def main():
    parser = argparse.ArgumentParser(description='Plot original curve')

    # parameter settings
    parser.add_argument('--q', type=int, default=1, help='numbering of question')
    parser.add_argument('--n_curve', type=int, default=100, help='samples to plot')

    # path settings
    parser.add_argument('--path_img', type=str, default='./images/', help='path of images')
    args = parser.parse_args()

    # original curve
    u = np.linspace(0, 1, args.n_curve, endpoint=True)
    x = 1.5 * (np.exp(1.5 * np.sin(6.2 * u - 0.027 * h)) + 0.1) * np.cos(12.2 * u)
    y = (np.exp(np.sin(6.2 * u - 0.027 * h)) + 0.1) * np.sin(12.2 * u)

    # plotting
    plt.figure()
    plt.plot(x, y, 'k--', label='Original curve', marker='.')
    plt.title('Curve r(u)')
    plt.legend(loc='best')
    plt.savefig(args.path_img + 'curve.png')
    plt.show()


if __name__ == '__main__':
    main()
