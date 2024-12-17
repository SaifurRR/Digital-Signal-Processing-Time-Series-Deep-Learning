{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The Parks-McClellan Filter Design Method\n",
    "\n",
    "The Parks-McClellan (PM) algorithm is a clever application of advanced polynomial fitting techniques to the problem of FIR filter design. In this notebook we will explore the key ideas behind the method by considering the design of a simple Type-I lowpass filter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "IPython.OutputArea.prototype._should_scroll = function(lines) {\n",
       "    return false;\n",
       "}"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%%javascript\n",
    "IPython.OutputArea.prototype._should_scroll = function(lines) {\n",
    "    return false;\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# standard bookkeeping\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from IPython.display import Audio, display\n",
    "from ipywidgets import interactive, fixed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "plt.rcParams[\"figure.figsize\"] = (14,4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "slideshow": {
     "slide_type": "slide"
    }
   },
   "source": [
    "**IMPORTANT** \n",
    "This notebook uses interactive widgets. If the interactive controls do not work or show up, please follow the instructions detailed here: https://github.com/jupyter-widgets/ipywidgets/blob/master/README.md"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The first intuition is to recognize that finding the filter's coefficients is equivalent to solving a polynomial fitting problem. Since we want a linear-phase filter, the filter will be symmetric and we can easily show that in this case its frequency response is the real-valued function:\n",
    "\n",
    "$$\n",
    "    H(e^{j\\omega}) = h[0] + \\sum_{n = 1}^{M} 2h[n]\\cos n\\omega\n",
    "$$\n",
    "\n",
    "In the above expression, the $N = 2M+1$ nonzero taps of the impulse response $h[n]$ are $h[-M]$ to $h[M]$. Given a (positive) passband $[0, \\omega_p]$ and a stopband $[\\omega_s, \\pi]$, we need to fit $H(e^{j\\omega})$ to one in the passband and zero in the stopband as in the following figure. \n",
    "\n",
    "<img src=\"specs.png\" alt=\"Drawing\" style=\"width: 600px;\"/>\n",
    "\n",
    "Enter [Chebishev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) (CP): they have a ton of interesting properties but the relevant one here is that $T_n(\\cos\\omega) = \\cos n\\omega$, where $T_n(x)$ is the CP of order $n$. Since the coefficients for the CP are easy to determine, with a simple substitution we can write:\n",
    "\n",
    "$$\n",
    "   H(e^{j\\omega}) =  \\sum_{n = 0}^{M}p_n x^n \\qquad \\mbox{with $x = \\cos\\omega$}\n",
    "$$\n",
    "\n",
    "While the relation between the $M+1$ $p_n$ coefficients and the $2M+1$ filter coefficients is nonlinear, it is easily invertible and therefore we can just concentrate on fitting the degree-$M$ polynomial over the desired response."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the sake of simplicity, let's skip some of the details associated to the Chebishev substitution (for instance, the mapping $x=\\cos\\omega$ reverses the abscissae) and let's examine the following equivalent problem:\n",
    "\n",
    "> consider the desired function: $$ D(x) = \\begin{cases} 1 & \\mbox{for $x\\in [0, A]$} \\\\ 0 & \\mbox{for $x\\in [B,1]$} \\end{cases}$$ <br> find the coefficients of a degree-$M$ polynomial $P(x)$ so that $P(x)\\approx D(x)$ over $[0,A]\\cup [B,1]$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Standard polynomial fitting\n",
    "\n",
    "The simplest approach is to use standard polynomial fitting: choose $a$ points in $[0, A]$ and $b$ points in $[B, 1]$ so that $a+b = M+1$ and find an interpolator over the coordinate pairs:\n",
    "\n",
    "$$\n",
    "  (x_0, 1), (x_1, 1), \\ldots, (x_{a-1}, 1), (x_a, 0), (x_{a+1}, 0), \\ldots, (x_{a+b-1}, 0)\n",
    "$$\n",
    "\n",
    "The result will minimize the mean square error between the interpolator and the piecewise characteristic we are trying to approximate. \n",
    "\n",
    "We can write a simple Python function to test this approach; you can play with the sliders below and see the result of the plain interpolation as the order and the size of the intervals change:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def MSE_fit(A, B, order):\n",
    "    if order < 3:\n",
    "        raise ValueError(\"order is too small, we need at least 3\")\n",
    "    \n",
    "    # interpolation points always one more than the order of the interpolator\n",
    "    pts = order+1\n",
    "    # split number of interpolation points across intervals proportionately \n",
    "    #  with the length of each interval \n",
    "    ptsA = int(pts * A / (A+(1-B)))\n",
    "    if ptsA < 2:\n",
    "        ptsA = 2\n",
    "    ptsB = pts - ptsA\n",
    "    \n",
    "    # for the MSE fit, place a point at each interval edge and distribute the rest\n",
    "    #  (if any) evenly over the interval\n",
    "    x = np.concatenate((\n",
    "        np.arange(0, ptsA) * (A / (ptsA-1)),\n",
    "        B + np.arange(0, ptsB) * ((1-B) / (ptsB-1))            \n",
    "        ))\n",
    "    y = np.concatenate((\n",
    "        np.ones(ptsA),\n",
    "        np.zeros(ptsB)            \n",
    "        ))\n",
    "    \n",
    "    # now just solve the linear interpolation problem\n",
    "    p = np.poly1d(np.polyfit(x, y, order))\n",
    "    return p, x, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def MSE_fit_show(A=0.4, B=0.6, order=10):\n",
    "    p, x, y = MSE_fit(A, B, order)\n",
    "    \n",
    "    t = np.linspace(0, 1, 100)\n",
    "    lims = [(0,1,-.5,1.5), (0,A,0.8,1.2), (B,1,-0.2,0.2)]\n",
    "    for n, r in enumerate(lims):\n",
    "        plt.subplot(1,3,n+1)\n",
    "        plt.plot((0,A), (1,1), 'red', \n",
    "                 (B,1), (0,0), 'red',  \n",
    "                 x, y, 'oy', \n",
    "                 t, p(t), '-')\n",
    "        plt.xlim(r[0], r[1])\n",
    "        plt.ylim(r[2], r[3]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "76fe6665a40d4775b14106114006d99c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/html": [
       "<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>\n",
       "<p>\n",
       "  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
       "  that the widgets JavaScript is still loading. If this message persists, it\n",
       "  likely means that the widgets JavaScript library is either not installed or\n",
       "  not enabled. See the <a href=\"https://ipywidgets.readthedocs.io/en/stable/user_install.html\">Jupyter\n",
       "  Widgets Documentation</a> for setup instructions.\n",
       "</p>\n",
       "<p>\n",
       "  If you're reading this message in another frontend (for example, a static\n",
       "  rendering on GitHub or <a href=\"https://nbviewer.jupyter.org/\">NBViewer</a>),\n",
       "  it may mean that your frontend doesn't currently support widgets.\n",
       "</p>\n"
      ],
      "text/plain": [
       "interactive(children=(FloatSlider(value=0.4, description='A', max=0.5), FloatSlider(value=0.6, description='B', max=1.0, min=0.5), IntSlider(value=10, description='order', max=30, min=3), Output()), _dom_classes=('widget-interact',))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "v = interactive(MSE_fit_show, order=(3,30), A=(0.0, 0.5), B=(0.5, 1.0))\n",
    "display(v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "As you can see, simple polynomial interpolation, while minimizing the MSE has two problems:\n",
    "\n",
    " * it becomes numerically unstable as soon as the order of the interpolation exceeds 16 or 17\n",
    " * although the MSE is minimized, the **maximum** error can become very large\n",
    " \n",
    "Because of these problems, direct interpolation is rarely used in numerical analysis and filter design is no exception. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Minimax fitting\n",
    "\n",
    "As we said, the first clever intuition behind the the Parks-McClellan algorithm is rephrasing the filter design problem as a polynomial interpolation. The real stroke of genius, however, is the use of a different kind of polynomial fitting called *minimax* approximation. In this kind of fitting the goal is to minimixe the *maximum* error between the polynomial and the desired function over the intervals of interest:\n",
    "\n",
    "$$\n",
    "    E = \\min\\max_{[0,A]\\cup [B,1]} | P(x) - D(x) |\n",
    "$$\n",
    "\n",
    "This is brilliant for two reasons: \n",
    "\n",
    "* the minimax criterion gives us a guarantee on the worst-case error for the filter response\n",
    "* an obscure yet powerful theorem, called the **alternation theorem**, gives us a remarkably straightforward recipe to build a robust numerical algorithm that finds the solution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The Alternation Theorem\n",
    "\n",
    "Because of the fundamental theorem of algebra, a non-constant polynomial cannot be constant over an interval. Since our polynomial cannot be a constant (it needs to move from 1 in the passband to zero in the stopband), it will necessarily oscillate over the approximation intervals. As you could see from the previous demo, MSE minimization tries to keep the oscillations small over the approximation intervals, but the price to pay is potentially wide oscillations at the band edges; on the other hand, the minimax approximation will allow for oscillations that are larger overall but that do not swing wildly. Since the polynomial oscillates around the target value, the error will oscillate between positive and negative peaks; the alternation theorem states that \n",
    "\n",
    "> $P(x)$ is the minimax approximation to $D(x)$ if and only if $P(x) - D(x)$ alternates $M+2$ times between $+E$ and $-E$ over $[0,A]\\cup [B,1]$\n",
    "\n",
    "The alternation theorem gives one thing right away: the ability to recognize if a polynomial is the minimax solution. All we need to do is look at the extrema of the error and check that \n",
    "\n",
    "* they are $M+2$\n",
    "* they alternate in sign\n",
    "* their magnitude is exactly the same\n",
    "\n",
    "We can write a simple Python function to find the extrema of the error:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def find_error_extrema(p, A, B):\n",
    "    intervals = {\n",
    "        (0, A): 1, \n",
    "        (B, 1): 0\n",
    "    }\n",
    "    loc = [] # locations of the extrema\n",
    "    err = [] # values of the extrema\n",
    "    for rng, val in intervals.items():\n",
    "        # we don't need enormous precision, 100 points per interval will do\n",
    "        t = np.linspace(rng[0], rng[1], 100)\n",
    "        y = val - p(t) # error values\n",
    "        # this finds all the points where the error changes sign:\n",
    "        ix = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 # local min+max\n",
    "        loc = np.hstack((loc, t[0], t[ix], t[-1]))\n",
    "        err = np.hstack((err, y[0], y[ix], y[-1]))\n",
    "    return loc, err"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With this, it's easy to verify that the MSE fit does not satisfy the alternation theorem: the magnitude of the peaks is not constant."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAzwAAAD8CAYAAACy9qHHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzs3Xd8VeXhx/HPc+/NzZ4kEAhhb2SHEW2dFVx1z7rqQsVRVx3tr9Xa/qrWWqmt4I8q1WrFUVvFKm4FW1lhyYYwQkISCITsdcfz+yOxjRBIgEtOxvf9euWVnHOee843Kabnm3POc421FhERERERkY7I5XQAERERERGRY0WFR0REREREOiwVHhERERER6bBUeEREREREpMNS4RERERERkQ5LhUdERERERDosFR4REREREemwVHhERERERKTDUuEREREREZEOy+N0gKYkJyfbPn36OB1DRERERETaqGXLlu2x1qY0N65NFp4+ffqQlZXldAwREREREWmjjDE5LRmnW9pERERERKTDUuEREREREZEOS4VHREREREQ6LBUeERERERHpsFR4RERERESkw1LhaaumTQOPB4yp/zxtmtOJRERERETanTY5LXWnN20azJz53+VA4L/LM2Y4k0lEREREpB3SFZ42Jjd7Nf7Zzze9cdas1g0jIiIiItLOqfC0ETYYZOFffkaPl7+Lp9bX9JhAgIDf38rJRERERETaLxWeNqCyvIQVvzuPzK3PsDLuZGqjwpscZ+INWx7PZE/hjlZOKCIiIiLSPqnwOCwvew1FT3+XUeVfsmjA3Yy9+++EX3v9AeOCxpB9xqn09OXge+5Utq/PciCtiIiIiEj7osLjoFWfvUHcK5OJD+5j3WkvMemqRzAuV/3EBLfeCm53/UC3G9cttzDgtU/YecFbhOEj6fXvs+bLd5z9BkRERERE2jhjrXU6wwEyMjJsVlbHvYIRDARY/PJPmbjtObZ5+hJ59Wv06DO4xa8v3LGZmhcvJC2wkxWjf8GEC+44hmlFRERERNoeY8wya21Gc+N0haeV+epqWfm788jcPpPl8afR454Fh1V2AFJ7DSTpzi/YEDGSCav+h4XP340NBo9RYhERERGR9kuFp5WteG8WYyu/ZGHf2xl315tERsce0X7iErow5N4PWZJ4Npl5s1k2/RJqa6pC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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1de1bb51828>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "A = 0.4\n",
    "B = 0.6\n",
    "p, x, y = MSE_fit(A, B, 8)\n",
    "\n",
    "loc, err = find_error_extrema(p, 0.4, 0.6)\n",
    "\n",
    "t = np.linspace(0, 1, 100)\n",
    "plt.plot(loc, p(loc), 'or', t, p(t), '-')\n",
    "plt.plot((0,A), (1,1), 'red', \n",
    "         (B,1), (0,0), 'red',  \n",
    "         t, p(t), '-',\n",
    "         loc, p(loc), 'or');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### The Remez Exchange Algorithm\n",
    "\n",
    "The alternation theorem provides us also with a very efficient way to find the coefficients of $P(x)$. Suppose we knew the exact ordinates $x_i$ of the $M+2$ alternations; in this case we could solve the following system of linear equations\n",
    "\n",
    "$$\n",
    "    \\left\\{\\begin{array}{lcl}\n",
    "        p_0 + p_1 x_0 + p_2 x_0^2 + \\ldots + p_Mx_0^M + (-1)^0\\epsilon &=& D(x_0) \\\\ \n",
    "        p_0 + p_1 x_1 + p_2 x_1^2 + \\ldots + p_Mx_1^M + (-1)^1\\epsilon &=& D(x_1) \\\\ \n",
    "        \\ldots \\\\ \n",
    "        p_0 + p_1 x_{M+1} + p_2 x_{M+1}^2 + \\ldots + p_Mx_{M+1}^M + (-1)^{M+1}\\epsilon &=& D(x_{M+1}) \n",
    "      \\end{array}\\right.\n",
    "$$\n",
    "\n",
    "and find in one go both the $M+1$ polynomial coefficients *and* the value of the minimax error $E=|\\epsilon|$ (we use $\\epsilon$ instead of $E$ in the linear system because we don't know the sign of the first alternation, it could be positive or negative). Although the above is a non-standard linear system of equations, it can be shown rather easily that, as long as the $x_i$ are distinct, it does have a solution. \n",
    "\n",
    "Of course we don't know the $x_i$ in advance but we can start with a guess, and solve the system anyway. Once we find the polynomial coefficients from the guess, we use Remez's exchange algorithm:\n",
    "\n",
    "* find the locations for the maxima and minima of the error for the $P(x)$ we just found\n",
    "* if the extrema satisfy the alternation theorem, we are done\n",
    "* otherwise, use the new locations as a new guess, solve the system again and repeat.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "The Remez algorithm is remarkably fast and robust. Here is an implementation you can play with. First, we need an auxiliary function to solve the system of equations above; we will use standard linear algebra functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def solve(x, y):\n",
    "    # simple solver for the extended interpolation problem\n",
    "    # first build a Vandermonde matrix using the interpolation locations\n",
    "    # There are N+2 locations, so the matrix will be (N+2)x(N+2) but we \n",
    "    #  will replace the last column with the error sign\n",
    "    V = np.vander(x, increasing=True)\n",
    "    # replace last column\n",
    "    V[:,-1] = pow(-1, np.arange(0, len(x)))\n",
    "    # just solve Ax = y\n",
    "    v = np.linalg.solve(V, y)\n",
    "    # need to reverse the vector because poly1d starts with the highest degree\n",
    "    p = np.poly1d(v[-2::-1])\n",
    "    e = np.abs(v[-1])\n",
    "    return p, e"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "And here comes the main course: the Remez routine. The code is quite straightforward; it doesn't have a termination conditions since the number of iterations is passed as a parameter (we want to be able to show intermediate results)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def remez_fit(A, B, order, iterations):\n",
    "    if order < 3:\n",
    "        raise ValueError(\"order is too small, we need at least 3\")\n",
    "    pts = order+2\n",
    "    \n",
    "    # initial choice of interpolation points: distribute them evenly\n",
    "    #  across the two regions as a proportion of each region's width\n",
    "    ptsA = int(pts * A / (A-B+1))\n",
    "    if ptsA < 2:\n",
    "        ptsA = 2\n",
    "    ptsB = pts - ptsA\n",
    "    \n",
    "    x = np.concatenate((\n",
    "        np.arange(1, ptsA+1) * (A / (ptsA+1)),\n",
    "        B + np.arange(1, ptsB+1) * ((1-B) / (ptsB+1))            \n",
    "        ))\n",
    "    y = np.concatenate((\n",
    "        np.ones(ptsA),\n",
    "        np.zeros(ptsB)            \n",
    "        ))\n",
    "    \n",
    "    # the \"data\" dictionary only holds values that we will use in plotting\n",
    "    data = {}\n",
    "    \n",
    "    for n in range(0, iterations):\n",
    "        # previous interpolation points\n",
    "        data['prev_x'] = x\n",
    "        data['prev_y'] = y\n",
    "        \n",
    "        # solve the interpolation problem \n",
    "        p, e = solve(x, y)\n",
    "        data['err'] = e\n",
    "        # find the extrema of the error\n",
    "        loc, err = find_error_extrema(p, A, B) \n",
    "        \n",
    "        # find the alternations \n",
    "        alt = []\n",
    "        for n in range(0, len(loc)):\n",
    "            # each extremum is a new candidate for an alternation\n",
    "            c = {\n",
    "                'loc': loc[n],\n",
    "                'sign': np.sign(err[n]),\n",
    "                'err_mag': np.abs(err[n])\n",
    "            }\n",
    "            # only keep extrema that are larger or equal than the minimum\n",
    "            #  error returned by the interpolation solution\n",
    "            if c['err_mag'] >= e - 1e-3:\n",
    "                # ensure that the error alternates; if the candidate has the  \n",
    "                #  same sign, replace the last alternation with the candidate\n",
    "                #  if its error value is larger\n",
    "                if alt == [] or alt[-1]['sign'] != c['sign']:\n",
    "                    alt.append(c)\n",
    "                elif alt[-1]['err_mag'] < c['err_mag']:\n",
    "                    alt.pop()\n",
    "                    alt.append(c)\n",
    "        \n",
    "        # if there are more than the necessary number of alternations, trim \n",
    "        #  from the left or the right keeping the largest errors\n",
    "        while len(alt) > order + 2:\n",
    "            if alt[0]['err_mag'] > alt[-1]['err_mag']:\n",
    "                alt.pop(-1)\n",
    "            else:\n",
    "                alt.pop(0)\n",
    "        \n",
    "        # the new set of interpolation points\n",
    "        x = [c['loc'] for c in alt]\n",
    "        y = [1 if c <= A else 0 for c in x]\n",
    "        data['new_x'] = x\n",
    "\n",
    "    return p, data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, a simple auxiliary function to plot the results; the yellow dots indicate the guess used for the current interpolation while the blue stars show the new maxima that will be used as the new guess. As you can see the algorithm converges very rapidly. The error in passband and stopband is shown magnified in the bottom panels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def remez_fit_show(A=0.4, B=0.6, order=5, iterations=1):\n",
    "    p, data = remez_fit(A, B, order, iterations)\n",
    "    t = np.linspace(0, 1, 200)\n",
    "    \n",
    "    def loc_plot(A, B, data):      \n",
    "        e = data['err']\n",
    "        plt.plot((0,A), (1,1), 'red',\n",
    "                 (B,1), (0,0), 'red', \n",
    "                 (0,A), (1+e,1+e), 'cyan', (0,A), (1-e,1-e), 'cyan',\n",
    "                 (B,1), (e,e), 'cyan', (B,1), (-e,-e), 'cyan',\n",
    "                 data['prev_x'], data['prev_y'], 'oy', \n",
    "                 data['new_x'], p(data['new_x']), '*',\n",
    "                 t, p(t), '-')  \n",
    "    \n",
    "    loc_plot(A, B, data)\n",
    "    plt.show()\n",
    "    \n",
    "    e = 1.5 * data['err']\n",
    "    lims = [(0, A , 1-e, 1+e), (B, 1, -e, e)]\n",
    "    for n, r in enumerate(lims):\n",
    "        plt.subplot(1,2,n+1)\n",
    "        loc_plot(A, B, data)\n",
    "        plt.xlim(r[0], r[1])\n",
    "        plt.ylim(r[2], r[3]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cb008e5a31284a1989108058c6480837",
       "version_major": 2,
       "version_minor": 0
      },
      "text/html": [
       "<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>\n",
       "<p>\n",
       "  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
       "  that the widgets JavaScript is still loading. If this message persists, it\n",
       "  likely means that the widgets JavaScript library is either not installed or\n",
       "  not enabled. See the <a href=\"https://ipywidgets.readthedocs.io/en/stable/user_install.html\">Jupyter\n",
       "  Widgets Documentation</a> for setup instructions.\n",
       "</p>\n",
       "<p>\n",
       "  If you're reading this message in another frontend (for example, a static\n",
       "  rendering on GitHub or <a href=\"https://nbviewer.jupyter.org/\">NBViewer</a>),\n",
       "  it may mean that your frontend doesn't currently support widgets.\n",
       "</p>\n"
      ],
      "text/plain": [
       "interactive(children=(FloatSlider(value=0.4, description='A', max=0.5, step=0.4), FloatSlider(value=0.6, description='B', max=1.0, min=0.5), IntSlider(value=5, description='order', max=12, min=3), IntSlider(value=1, description='iterations', max=10, min=1), Output()), _dom_classes=('widget-interact',))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "v = interactive(remez_fit_show, A=(0.0, 0.5, 0.4), B=(0.5, 1.0), order=(3,12), iterations=(1, 10))\n",
    "display(v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Weighting the error\n",
    "\n",
    "In the example above we used a very simple piecewise constant target function $D(x)$ but the Alternation Theorem is actually much more general and the Remez algorithm can be used to approximate more complicated characteristics. \n",
    "\n",
    "In particular, a *weighting function* can be used in the minimax problem formulation in order to tune the error in the approximation regions. Think of minimax filter design as a budgeting problem: for a given number of coefficients, we need to make sure that the bandpass and bandstop reqirements are fulfilled first, while the error is a consequence of the budget. With a weighting function we can decide to allocate a larger part of the budget to either passband or stopband. \n",
    "\n",
    "Mathematically, the minimax problem becomes:\n",
    "\n",
    "$$\n",
    "    E = \\min\\max_{[0,A]\\cup [B,1]} | W(x)[P(x) - D(x)] |\n",
    "$$\n",
    "\n",
    "but the alternation theorem still holds:\n",
    "\n",
    "> $P(x)$ is the minimax approximation to $D(x)$ *weighted by $W(x)$* if and only if $W(x)[P(x) - D(x)]$ alternates $M+2$ times between $+E$ and $-E$ over $[0,A]\\cup [B,1]$\n",
    "\n",
    "For instance, suppose we want the error in the first interval to be 10 times smaller than in the second interval. In this case the weighting function will be equal to 0.1 over $[0,A]$ and just one over $[B,1]$. We can rewrite the extended interpolation problem as\n",
    "\n",
    "$$\n",
    "    \\left\\{\\begin{array}{lcl}\n",
    "        p_0 + p_1 x_0 + p_2 x_0^2 + \\ldots + p_Mx_0^M + (-1)^0\\epsilon/10 &=& 1 \\\\ \n",
    "        p_0 + p_1 x_1 + p_2 x_1^2 + \\ldots + p_Mx_1^M + (-1)^1\\epsilon/10 &=& 1 \\\\ \n",
    "        \\ldots \\\\ \n",
    "        p_0 + p_1 x_{M} + p_2 x_{M}^2 + \\ldots + p_Mx_{M}^M + (-1)^{M}\\epsilon &=& 0\\\\ \n",
    "        p_0 + p_1 x_{M+1} + p_2 x_{M+1}^2 + \\ldots + p_Mx_{M+1}^M + (-1)^{M+1}\\epsilon &=& 0 \n",
    "      \\end{array}\\right.\n",
    "$$\n",
    "\n",
    "The following code is a simple modification of the algorithm detailed above to include error weighting:\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def remez_fit2(A, B, Aweight, order, iterations):\n",
    "    def weigh(x):\n",
    "        # the weighting function\n",
    "        if np.isscalar(x):\n",
    "            return 1.0/Aweight if x <= A else 1\n",
    "        else:\n",
    "            return [1.0/Aweight if v <= A else 1 for v in x]\n",
    "        \n",
    "    pts = order+2\n",
    "    ptsA = int(pts * A / (A-B+1))\n",
    "    if ptsA < 2:\n",
    "        ptsA = 2\n",
    "    ptsB = pts - ptsA\n",
    "    x = np.concatenate((\n",
    "        np.arange(1, ptsA+1) * (A / (ptsA+1)),\n",
    "        B + np.arange(1, ptsB+1) * ((1-B) / (ptsB+1))            \n",
    "        ))\n",
    "    y = np.concatenate((\n",
    "        np.ones(ptsA),\n",
    "        np.zeros(ptsB)            \n",
    "        ))\n",
    "    \n",
    "    data = {}    \n",
    "    for n in range(0, iterations):\n",
    "        data['prev_x'] = x\n",
    "        data['prev_y'] = y\n",
    "        \n",
    "        # solve the interpolation problem with weighted error\n",
    "        V = np.vander(x, increasing=True)\n",
    "        V[:,-1] = pow(-1, np.arange(0, len(x))) * weigh(x)\n",
    "        v = np.linalg.solve(V, y)\n",
    "        p = np.poly1d(v[-2::-1])\n",
    "        e = np.abs(v[-1])\n",
    "        data['Aerr'] = e / Aweight\n",
    "        data['Berr'] = e\n",
    "        \n",
    "        loc, err = find_error_extrema(p, A, B) \n",
    "        alt = []\n",
    "        for n in range(0, len(loc)):\n",
    "            c = {\n",
    "                'loc': loc[n],\n",
    "                'sign': np.sign(err[n]),\n",
    "                'err_mag': np.abs(err[n]) / weigh(loc[n])\n",
    "            }\n",
    "            if c['err_mag'] >= e - 1e-3:\n",
    "                if alt == [] or alt[-1]['sign'] != c['sign']:\n",
    "                    alt.append(c)\n",
    "                elif alt[-1]['err_mag'] < c['err_mag']:\n",
    "                    alt.pop()\n",
    "                    alt.append(c)\n",
    "        while len(alt) > order + 2:\n",
    "            if alt[0]['err_mag'] > alt[-1]['err_mag']:\n",
    "                alt.pop(-1)\n",
    "            else:\n",
    "                alt.pop(0)\n",
    "        \n",
    "        x = [c['loc'] for c in alt]\n",
    "        y = [1 if c <= A else 0 for c in x]\n",
    "        data['new_x'] = x\n",
    "\n",
    "    return p, data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "807e40bd3ca44081abcd003fc5838d39",
       "version_major": 2,
       "version_minor": 0
      },
      "text/html": [
       "<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>\n",
       "<p>\n",
       "  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
       "  that the widgets JavaScript is still loading. If this message persists, it\n",
       "  likely means that the widgets JavaScript library is either not installed or\n",
       "  not enabled. See the <a href=\"https://ipywidgets.readthedocs.io/en/stable/user_install.html\">Jupyter\n",
       "  Widgets Documentation</a> for setup instructions.\n",
       "</p>\n",
       "<p>\n",
       "  If you're reading this message in another frontend (for example, a static\n",
       "  rendering on GitHub or <a href=\"https://nbviewer.jupyter.org/\">NBViewer</a>),\n",
       "  it may mean that your frontend doesn't currently support widgets.\n",
       "</p>\n"
      ],
      "text/plain": [
       "interactive(children=(FloatSlider(value=0.4, description='A', max=0.5, step=0.4), FloatSlider(value=0.6, description='B', max=1.0, min=0.5), IntSlider(value=50, description='Aweight', min=1, step=10), IntSlider(value=10, description='order', max=20, min=5), IntSlider(value=1, description='iterations', max=10, min=1), Output()), _dom_classes=('widget-interact',))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def remez_fit_show2(A=0.4, B=0.6, Aweight=50, order=10, iterations=1):\n",
    "    p, data = remez_fit2(A, B, Aweight, order, iterations)\n",
    "    \n",
    "    t = np.linspace(0, 1, 300)\n",
    "    Ae = data['Aerr']\n",
    "    Be = data['Berr']\n",
    "    \n",
    "    def loc_plot(A, B, data):  \n",
    "        plt.plot((0,A), (1,1), 'red',\n",
    "                 (B,1), (0,0), 'red', \n",
    "                 (0,A), (1+Ae,1+Ae), 'cyan', (0,A), (1-Ae,1-Ae), 'cyan',\n",
    "                 (B,1), (Be,Be), 'cyan', (B,1), (-Be,-Be), 'cyan',\n",
    "                 data['prev_x'], data['prev_y'], 'oy', \n",
    "                 data['new_x'], p(data['new_x']), '*',\n",
    "                 t, p(t), '-')  \n",
    "    \n",
    "    loc_plot(A, B, data)\n",
    "    plt.show()\n",
    "    \n",
    "    lims = [(0, A , 1-1.5*Ae, 1+1.5*Ae), (B, 1, -1.5*Be, 1.5*Be)]\n",
    "    for n, r in enumerate(lims):\n",
    "        plt.subplot(1,2,n+1)\n",
    "        loc_plot(A, B, data)\n",
    "        plt.xlim(r[0], r[1])\n",
    "        plt.ylim(r[2], r[3]) \n",
    "        \n",
    "        \n",
    "        \n",
    "v = interactive(remez_fit_show2, \n",
    "                A=(0.0, 0.5, 0.4), B=(0.5, 1.0), \n",
    "                Aweight=(1,100,10),\n",
    "                order=(5,20), \n",
    "                iterations=(1, 10))\n",
    "display(v)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}