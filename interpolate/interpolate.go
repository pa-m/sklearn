package interpolate

import (
	"fmt"
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
func (s *xy) XY(i int) (float64, float64) { return s.x[i], s.y[i] }

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
	if len(x) < 2 || len(x) != len(y) {
		panic(fmt.Errorf("interp1d lenx:%d leny:%d", len(x), len(y)))
	}

	var both *xy
	if sort.Float64sAreSorted(x) {
		both = &xy{x, y}
	} else {
		both = &xy{make([]float64, len(x)), make([]float64, len(x))}
		copy(both.x, x)
		copy(both.y, y)
		sort.Sort(both)
	}
	return func(x float64) float64 {
		ix := sort.SearchFloat64s(both.x, x) - 1
		if ix < 0 {
			ix = 0
		}
		if ix > len(both.x)-2 {
			ix = len(both.x) - 2
		}
		ix1 := ix + 1
		for ix > 0 && math.IsNaN(both.y[ix]) {
			ix--
		}
		for ix1 < len(both.x)-1 && math.IsNaN(both.y[ix1]) {
			ix1++
		}
		//fmt.Println(x, both.x, both.y, ix, ix1)
		return interpolate2points(both.x[ix], both.y[ix], both.x[ix1], both.y[ix1])(x)
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
			if xsorted[i] > xsorted[ix0] && xsorted[i] <= x && xsorted[i] <= xsorted[ix2] {
				ix0 = i
			}
			if xsorted[l-1-i] <= xsorted[ix2] && xsorted[l-1-i] >= x && xsorted[l-1-i] > xsorted[ix0] {
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
		x2 := xsorted[ix2]
		for ix2 > 1 && xsorted[ix2-1] == x2 {
			ix2--
		}
		zxlow := Interp1d(ysorted[ix0:ix1], zsorted[ix0:ix1])(y)
		zxhi := Interp1d(ysorted[ix2:ix3], zsorted[ix2:ix3])(y)
		//fmt.Println("xlow", xsorted[ix0:ix1], "xhi", xsorted[ix2:ix3], "yxlow", ysorted[ix0:ix1], "yxhi", ysorted[ix2:ix3], "zxlow", zsorted[ix0:ix1], zxlow, "zxhi", zsorted[ix2:ix3], zxhi)
		return Interp1d([]float64{xsorted[ix0], xsorted[ix2]}, []float64{zxlow, zxhi})(x)
	}
}
