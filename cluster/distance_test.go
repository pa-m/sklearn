package cluster

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestMinkowskiDistanceP(t *testing.T) {
	a := mat.NewVecDense(2, []float64{0, 0})
	b := mat.NewVecDense(2, []float64{1, 2})
	if MinkowskiDistanceP(a, a, 2) != 0 {
		t.Fail()
	}
	if MinkowskiDistanceP(b, b, 2) != 0 {
		t.Fail()
	}
	if math.Abs(5-MinkowskiDistanceP(a, b, 2)) > 1e-9 {
		t.Fail()
	}
	actual := MinkowskiDistanceP(a, b, math.Inf(1))
	if math.Abs(2-actual) > 1e-9 {
		t.Errorf("got %g", actual)
	}
}
