# /user/bin/env python
# __*__ utf-8 __*__

import argparse
import numpy as np 
from scipy import special, optimize
import matplotlib.pyplot as plt 

def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--order", type=int, default=3, help="order of Bessel function")
    parser.add_argument("--output", default="plot.png", help="output image file")
    args = parser.parse_args()

    f = lambda x: -special.jv(args.order, x)
    sol = optimize.minimize(f, 1.0)

    x = np.linspace(0, 10, 5000)
    plt.plot(x, special.jv(args.order, x), '-', sol.x, -sol.fun, 'o')
    plt.savefig(args.output, dpi=96)
    print("in main function")
if __name__ == "__main__":
    main()