package interpolate

import (
	"math"
	"sort"
)

type xy struct{ x, y []float64 }

func (s *xy) Len() int           { return len(s.x) }
func (s *xy) Less(i, j int) bool { return s.x[i] < s.x[j] }
func (s *xy) Swap(i, j int) {
	s.x[i], s.x[j] = s.x[j], s.x[i]
	s.y[i], s.y[j] = s.y[j], s.y[i]
}

func interpolate2points(x0, y0, x1, y1 float64) func(float64) float64 {
	return func(x float64) float64 {
		if x1 == x0 {
			return (y0 + y1) / 2.
		}
		return y0 + (x-x0)/(x1-x0)*(y1-y0)
	}
}

// Interp1d return linear interpolation from x,y points
// mimics partly scipy.interpolate.interp1d
func Interp1d(x, y []float64) func(x float64) float64 {
	xsorted, ysorted := make([]float64, len(x)), make([]float64, len(x))
	copy(xsorted, x)
	copy(ysorted, y)
	both := &xy{x: xsorted, y: ysorted}
	sort.Sort(both)
	return func(x float64) float64 {
		for ix := range xsorted[:len(xsorted)-1] {
			if x < xsorted[ix+1] || ix == len(xsorted)-2 {
				return interpolate2points(xsorted[ix], ysorted[ix], xsorted[ix+1], ysorted[ix+1])(x)
			}
		}
		return math.NaN()
	}
}

type xyz struct{ x, y, z []float64 }

func (s *xyz) Len() int           { return len(s.x) }
func (s *xyz) Less(i, j int) bool { return s.x[i] < s.x[j] || (s.x[i] == s.x[j] && s.y[i] < s.y[j]) }
func (s *xyz) Swap(i, j int) {
	s.x[i], s.x[j] = s.x[j], s.x[i]
	s.y[i], s.y[j] = s.y[j], s.y[i]
	s.z[i], s.z[j] = s.z[j], s.z[i]
}

// Interp2d computes bilinear interpolation from x,y to z
func Interp2d(x, y, z []float64) func(x, y float64) float64 {
	xsorted, ysorted, zsorted := make([]float64, len(x)), make([]float64, len(x)), make([]float64, len(x))
	copy(xsorted, x)
	copy(ysorted, y)
	copy(zsorted, z)
	all := &xyz{x: xsorted, y: ysorted, z: zsorted}
	sort.Sort(all)
	return func(x, y float64) float64 {
		ix0 := 0
		l := len(xsorted)
		ix2 := l - 1
		for i := range xsorted {
			if xsorted[i] > xsorted[ix0] && xsorted[i] <= x {
				ix0 = i
			}
			if xsorted[l-1-i] <= xsorted[ix2] && xsorted[l-1-i] >= x {
				ix2 = l - 1 - i
			}
		}
		ix1 := ix0
		for ix1 < l && xsorted[ix1] == xsorted[ix0] {
			ix1++
		}
		ix3 := ix2
		for ix3 < l && xsorted[ix3] == xsorted[ix2] {
			ix3++
		}
		zxlow := Interp1d(ysorted[ix0:ix1], zsorted[ix0:ix1])(y)
		zxhi := Interp1d(ysorted[ix2:ix3], zsorted[ix2:ix3])(y)
		//fmt.Println("xlow", xsorted[ix0], zxlow, "xhi", xsorted[ix2], zxhi)
		return Interp1d([]float64{xsorted[ix0], xsorted[ix2]}, []float64{zxlow, zxhi})(x)
	}
}
