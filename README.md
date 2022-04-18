## Interpolation / fitting methods

This is a implementation of several common interpolation / fitting methods.

---
### Experiment

1. Plot original curve for experiment
    ```bash
    python main_plot.py
    ```

2. Fit a parametric cubic polynomial curve to original curve using least-squares method
    ```bash
    python main_least_square.py
    ```

3. Interpolate samples from original curve using cubic B-spline interpolation algorithm
    ```bash
    python main_b_spline.py
    ```

4. Approximate original curve using Discrete Fourier Transform and trigonometric interpolation
    ```bash
    python main_trigonometric_interpolate.py
    ```

5. Compute integral of 1 component of original curve using composite Simpson’s rule
    ```bash
    python main_composite_simpson.py
    ```