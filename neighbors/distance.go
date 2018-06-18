package neighbors

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Distance has Distance(Vector,Vector)float64
type Distance func(a, b mat.Vector) float64

// MinkowskiDistanceP ...
func MinkowskiDistanceP(a, b mat.Vector, p float64) float64 {
	if a.Len() == 1 {
		return math.Abs(b.At(0, 0) - a.At(0, 0))
	}
	var dp float64
	rva, isrva := a.(mat.RawVectorer)
	rvb, isrvb := b.(mat.RawVectorer)
	if math.IsInf(p, 1) {

		if isrva && isrvb {
			for araw, braw, j := rva.RawVector().Data, rvb.RawVector().Data, 0; j < len(araw); j++ {
				dp = math.Max(dp, math.Abs(braw[j]-araw[j]))
			}
		} else {
			for j := 0; j < b.Len(); j++ {
				dp = math.Max(dp, math.Abs(b.AtVec(j)-a.AtVec(j)))
			}
		}
		return dp
	}
	if isrva && isrvb {
		for araw, braw, j := rva.RawVector().Data, rvb.RawVector().Data, 0; j < len(araw); j++ {
			dp += math.Pow(math.Abs(braw[j]-araw[j]), p)
		}
	} else {
		for j := 0; j < b.Len(); j++ {
			dp += math.Pow(math.Abs(b.AtVec(j)-a.AtVec(j)), p)
		}
	}
	return dp
}

// MinkowskiDistance ...
func MinkowskiDistance(p float64) Distance {

	return func(a, b mat.Vector) float64 {
		d := MinkowskiDistanceP(a, b, p)
		if !math.IsInf(p, 1) && p != 1. {
			d = math.Pow(d, 1./p)
		}
		return d

	}
}

// EuclideanDistance is a Distancer
func EuclideanDistance(a, b mat.Vector) float64 {
	if a.Len() == 1 {
		return math.Abs(b.At(0, 0) - a.At(0, 0))
	}
	var d2 float64
	avd, avdok := a.(mat.RawVectorer)
	bvd, bvdok := b.(mat.RawVectorer)
	if avdok && bvdok {
		arv, brv, vb, d := avd.RawVector(), bvd.RawVector(), 0., 0.
		for j, va := range arv.Data {
			vb = brv.Data[j]
			d = vb - va
			d2 += d * d
		}
	} else {
		for j := 0; j < b.Len(); j++ {
			x := b.AtVec(j) - a.AtVec(j)
			d2 += x * x
		}

	}
	return math.Sqrt(d2)
}
