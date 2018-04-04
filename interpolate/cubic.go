package interpolate

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// CubicSpline adapted from https://github.com/morganherlocker/cubic-spline
func CubicSpline(xs, ys []float64) func(x float64) float64 {

	swapRows := func(m *mat.Dense, k, l int) {
		rm := m.RawMatrix()
		for j := 0; j < rm.Cols; j++ {
			rm.Data[k*rm.Stride+j], rm.Data[l*rm.Stride+j] = rm.Data[l*rm.Stride+j], rm.Data[k*rm.Stride+j]
		}
	}
	solve := func(A *mat.Dense, ks []float64) []float64 {
		var m, _ = A.Dims()
		for k := 0; k < m; k++ { // column

			// pivot for column
			var iMax = 0
			var vali = math.Inf(-1)
			for i := k; i < m; i++ {
				if A.At(i, k) > vali {
					iMax = i
					vali = A.At(i, k)
				}
			}
			swapRows(A, k, iMax)

			// for all rows below pivot
			for i := k + 1; i < m; i++ {
				for j := k + 1; j < m+1; j++ {
					A.Set(i, j, A.At(i, j)-A.At(k, j)*(A.At(i, k)/A.At(k, k)))
				}
				A.Set(i, k, 0.)
			}
		}
		for i := m - 1; i >= 0; i-- { // rows = columns

			var v = A.At(i, m) / A.At(i, i)
			ks[i] = v
			for j := i - 1; j >= 0; j-- { // rows

				A.Set(j, m, A.At(j, m)-A.At(j, i)*v)
				A.Set(j, i, 0.)
			}
		}
		return ks
	}

	getNaturalKs := func(xs, ys, ks []float64) []float64 {
		var n = len(xs) - 1
		var A = mat.NewDense(n+1, n+2, nil)

		for i := 1; i < n; i++ { // rows

			A.Set(i, i-1, 1./(xs[i]-xs[i-1]))
			A.Set(i, i, 2.*(1/(xs[i]-xs[i-1])+1/(xs[i+1]-xs[i])))
			A.Set(i, i+1, 1./(xs[i+1]-xs[i]))
			A.Set(i, n+1, 3.*((ys[i]-ys[i-1])/((xs[i]-xs[i-1])*(xs[i]-xs[i-1]))+(ys[i+1]-ys[i])/((xs[i+1]-xs[i])*(xs[i+1]-xs[i]))))
		}

		A.Set(0, 0, 2./(xs[1]-xs[0]))
		A.Set(0, 1, 1./(xs[1]-xs[0]))
		A.Set(0, n+1, 3.*(ys[1]-ys[0])/((xs[1]-xs[0])*(xs[1]-xs[0])))

		A.Set(n, n-1, 1./(xs[n]-xs[n-1]))
		A.Set(n, n, 2./(xs[n]-xs[n-1]))
		A.Set(n, n+1, 3.*(ys[n]-ys[n-1])/((xs[n]-xs[n-1])*(xs[n]-xs[n-1])))

		return solve(A, ks)
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
