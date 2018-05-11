package neighbors

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Distance has Distance(Vector,Vector)float64
type Distance func(a, b mat.Vector) float64

// MinkowskiDistance ...
func MinkowskiDistance(p float64) Distance {
	return func(a, b mat.Vector) float64 {
		if a.Len() == 1 {
			return math.Abs(b.At(0, 0) - a.At(0, 0))
		}
		var dp float64
		for j := 0; j < b.Len(); j++ {
			dp += math.Pow(math.Abs(b.AtVec(j)-a.AtVec(j)), p)
		}
		return math.Pow(dp, 1./p)
	}
}

// EuclideanDistance is a Distancer
func EuclideanDistance(a, b mat.Vector) float64 {
	if a.Len() == 1 {
		return math.Abs(b.At(0, 0) - a.At(0, 0))
	}
	var d2 float64
	for j := 0; j < b.Len(); j++ {
		d2 += math.Pow(b.AtVec(j)-a.AtVec(j), 2)
	}
	return math.Sqrt(d2)
}
