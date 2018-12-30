package neuralnetwork

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAbsSum(t *testing.T) {
	m := mat.NewDense(2, 4, []float64{1, -1, 2, -2, -3, 3, -4, 4})
	if 20 != (&matx{Dense: m}).SumAbs() {
		t.Fail()
	}
	actual := (&matx{Dense: m.Slice(0, 2, 1, 4).(*mat.Dense)}).SumAbs()
	if 16 != actual {
		fmt.Println(mat.Formatted(m.Slice(0, 2, 1, 4).(*mat.Dense)))
		t.Errorf("expected 18, got %g", actual)
	}
}

func TestSumSquares(t *testing.T) {
	m := mat.NewDense(2, 4, []float64{1, -1, 2, -2, -3, 3, -4, 4})
	if actual := (&matx{Dense: m}).SumSquares(); 60 != actual {
		t.Errorf("expected 60, got %g", actual)
	}
	actual := (&matx{Dense: m.Slice(0, 2, 1, 4).(*mat.Dense)}).SumSquares()
	if 50 != actual {
		fmt.Println(mat.Formatted(m.Slice(0, 2, 1, 4).(*mat.Dense)))
		t.Errorf("expected 18, got %g", actual)
	}
}
