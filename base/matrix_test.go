package base

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMatrixFirstColumnRemoved(t *testing.T) {
	Theta := mat.NewDense(1, 3, []float64{1, 2, 3})
	Theta1 := MatFirstColumnRemoved{Theta}
	_, c := Theta.Dims()
	_, c1 := Theta1.Dims()
	if c1 != c-1 {
		t.Errorf("c1:%d expected:%d", c1, c-1)
	}
	if Theta1.At(0, 0) != 2 {
		t.Errorf("Theta1.At(0,0):%g expected:%g", Theta1.At(0, 0), 2.)
	}
}
