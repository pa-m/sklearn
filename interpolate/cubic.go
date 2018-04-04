package interpolate

import (
	"sort"

	"gonum.org/v1/gonum/mat"
)

// CubicSpline adapted from https://github.com/morganherlocker/cubic-spline
func CubicSpline(xs, ys []float64) func(x float64) float64 {

	getNaturalKs := func(xs, ys, ks []float64) []float64 {
		var n = len(xs) - 1
		var Adense = mat.NewDense(n+1, n+2, nil)
		A := Adense.RawMatrix()

		for i, ai := 1, A.Stride; i < n; i, ai = i+1, ai+A.Stride { // rows
			A.Data[ai+i-1] = 1. / (xs[i] - xs[i-1])
			A.Data[ai+i] = 2. * (1/(xs[i]-xs[i-1]) + 1/(xs[i+1]-xs[i]))
			A.Data[ai+i+1] = 1. / (xs[i+1] - xs[i])
			// bias at line i
			A.Data[ai+n-1] = 3. * ((ys[i]-ys[i-1])/((xs[i]-xs[i-1])*(xs[i]-xs[i-1])) + (ys[i+1]-ys[i])/((xs[i+1]-xs[i])*(xs[i+1]-xs[i])))
		}
		ai := 0
		A.Data[ai+0] = 2. / (xs[1] - xs[0])
		A.Data[ai+1] = 1. / (xs[1] - xs[0])
		// bias at line 0
		A.Data[ai+n+1] = 3. * (ys[1] - ys[0]) / ((xs[1] - xs[0]) * (xs[1] - xs[0]))

		ai = n * A.Stride
		A.Data[ai+n-1] = 1. / (xs[n] - xs[n-1])
		A.Data[ai+n] = 2. / (xs[n] - xs[n-1])
		// bias at line n
		A.Data[ai+n+1] = 3. * (ys[n] - ys[n-1]) / ((xs[n] - xs[n-1]) * (xs[n] - xs[n-1]))

		ksMat := mat.NewDense(len(ks), 1, ks)
		ksMat.Solve(Adense.Slice(0, n+1, 0, n+1), Adense.Slice(0, n+1, n+1, n+2))

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
