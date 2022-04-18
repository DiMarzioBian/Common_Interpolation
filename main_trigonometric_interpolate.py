import argparse
import numpy as np
from numpy import pi as pi
from scipy import interpolate
import matplotlib.pyplot as plt

h = 3


def main():
    parser = argparse.ArgumentParser(description='Approximate curve using DFT and trigonometric interpolation')

    # parameter settings
    parser.add_argument('--alpha_base', type=int, default=1, help='multiple of frequency to standard base')
    parser.add_argument('--n_base', type=int, default=8, help='number of base function')
    parser.add_argument('--n_curve', type=int, default=100, help='number of point to plot curve')
    parser.add_argument('--n_sample', type=int, default=15, help='number of point before interpolation')

    # path settings
    parser.add_argument('--path_img', type=str, default='./images/', help='path of images')
    args = parser.parse_args()

    # original curve
    u_curve = np.linspace(0, 1, args.n_curve, endpoint=True)
    x_curve = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_curve - 0.027 * h)) + 0.1) * np.cos(12.2 * u_curve)
    y_curve = (np.exp(np.sin(6.2 * u_curve - 0.027 * h)) + 0.1) * np.sin(12.2 * u_curve)

    # sampling
    u_sample = np.linspace(0, 1, args.n_sample, endpoint=False)
    x_sample = 1.5 * (np.exp(1.5 * np.sin(6.2 * u_sample - 0.027 * h)) + 0.1) * np.cos(12.2 * u_sample)
    y_sample = (np.exp(np.sin(6.2 * u_sample - 0.027 * h)) + 0.1) * np.sin(12.2 * u_sample)

    # fitting
    x_fft = np.fft.fft(x_sample)
    y_fft = np.fft.fft(y_sample)

    A_x = np.array([i.real for i in x_fft])
    B_x = np.array([i.imag for i in x_fft])
    A_y = np.array([i.real for i in y_fft])
    B_y = np.array([i.imag for i in y_fft])

    K = args.n_base // 2
    N = args.n_sample
    x_fit = A_x[0] / N + A_x[K] * np.cos(pi * u_sample * N) / N
    x_fit += sum([A_x[k] * np.cos(args.alpha_base * pi * u_sample * k)
                  - B_x[k] * np.sin(args.alpha_base * pi * u_sample * k) for k in range(1, K)]) * 2 / N

    y_fit = A_y[0] / N + A_y[K] * np.cos(pi * u_sample * N) / N
    y_fit += sum([A_y[k] * np.cos(args.alpha_base * pi * u_sample * k)
                  - B_y[k] * np.sin(args.alpha_base * pi * u_sample * k) for k in range(1, K)]) * 2 / N

    # plotting
    plt.figure()
    plt.plot(x_curve, y_curve, 'k--', label='Original curve', marker='.')
    plt.plot(x_sample, y_sample, 'b--', label='Sampled curve', marker='.')
    plt.plot(x_fit, y_fit, label='Fit curve', marker='.', color='red')
    plt.legend(loc='best')
    plt.title('Trigonometric interpolation with {n_base} bases on {n_sample} samples'
              .format(n_base=args.n_base, n_sample=args.n_sample))
    plt.savefig(args.path_img + 'trigonometric_' + str(args.n_sample) + '_' + str(args.n_base) + '.png')
    plt.show()


if __name__ == '__main__':
    main()
