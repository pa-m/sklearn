package interpolate

import (
	"errors"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// CubicSpline adapted from https://github.com/morganherlocker/cubic-spline
func CubicSpline(xs, ys []float64) func(x float64) float64 {

	getNaturalKs := func(xs, ys, ks []float64) []float64 {
		var n = len(xs) - 1
		var A = mat.NewSymBandDense(n+1, 1, nil)
		Bd := mat.NewVecDense(n+1, nil)

		A.SetSymBand(0, 0, 2./(xs[1]-xs[0]))
		A.SetSymBand(0, 1, 1./(xs[1]-xs[0]))
		// bias at line 0
		Bd.SetVec(0, 3.*(ys[1]-ys[0])/((xs[1]-xs[0])*(xs[1]-xs[0])))

		for i := 1; i < n; i++ { // rows
			A.SetSymBand(i, i-1, 1./(xs[i]-xs[i-1]))
			A.SetSymBand(i, i, 2.*(1/(xs[i]-xs[i-1])+1/(xs[i+1]-xs[i])))
			A.SetSymBand(i, i+1, 1./(xs[i+1]-xs[i]))
			// bias at line i
			Bd.SetVec(i, 3.*((ys[i]-ys[i-1])/((xs[i]-xs[i-1])*(xs[i]-xs[i-1]))+(ys[i+1]-ys[i])/((xs[i+1]-xs[i])*(xs[i+1]-xs[i]))))
		}

		A.SetSymBand(n, n-1, 1./(xs[n]-xs[n-1]))
		A.SetSymBand(n, n, 2./(xs[n]-xs[n-1]))
		// bias at line n
		Bd.SetVec(n, 3.*(ys[n]-ys[n-1])/((xs[n]-xs[n-1])*(xs[n]-xs[n-1])))

		ksVec := mat.NewVecDense(len(ks), ks)

		var err error
		var chol mat.Cholesky
		if chol.Factorize(A) {
			err = chol.SolveVecTo(ksVec, Bd)
		} else {
			err = errors.New("not positive definite")
		}

		if err != nil {
			ksVec.SolveVec(A, Bd)
		}

		return ks

	}

	var both *xy
	if sort.Float64sAreSorted(xs) {
		both = &xy{xs, ys}
	} else {
		both = &xy{make([]float64, len(xs)), make([]float64, len(xs))}
		copy(both.x, xs)
		copy(both.y, ys)
		sort.Sort(both)
	}

	var ks = make([]float64, len(xs))
	ks = getNaturalKs(both.x, both.y, ks)
	return func(x float64) float64 {
		var i = 1
		for i < len(both.x)-1 && both.x[i] < x {
			i++
		}
		var t = (x - both.x[i-1]) / (both.x[i] - both.x[i-1])
		var a = ks[i-1]*(both.x[i]-both.x[i-1]) - (both.y[i] - both.y[i-1])
		var b = -ks[i]*(both.x[i]-both.x[i-1]) + (both.y[i] - both.y[i-1])
		var q = (1-t)*both.y[i-1] + t*both.y[i] + t*(1-t)*(a*(1-t)+b*t)
		return q
	}
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
