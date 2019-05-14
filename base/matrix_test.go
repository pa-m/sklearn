package base

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

var _ mat.Mutable = MatTranspose{}

func TestMatConst(t *testing.T) {
	m := MatConst{Rows: 2, Columns: 3, Value: 1}
	r, c := m.Dims()
	if r != 2 || c != 3 {
		t.Fail()
	}
	if m.At(0, 0) != 1 || m.At(1, 1) != 1 {
		t.Fail()
	}
	if m.T().At(0, 0) != 1 {
		t.Fail()
	}
}
func TestMatTranspose(t *testing.T) {
	d := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	m := MatTranspose{Matrix: d}
	if m.At(1, 0) != 1 {
		t.Fail()
	}
	m.Set(1, 0, 3)
	if m.T().At(0, 1) != 3 {
		t.Fail()
	}
}

func TestMatStr(t *testing.T) {
	actual := MatStr(mat.NewDense(2, 2, []float64{1, 2, 4, 5}), mat.NewDense(2, 1, []float64{3, 6}))
	if MatStr() != "" {
		t.Fail()
	}
	if MatStr(&mat.Dense{}) != "" {
		t.Fail()
	}
	expected := "1\t2\t3\n4\t5\t6\n"
	if expected != actual {
		t.Errorf("expected:\n%s\ngot:\n%s\n", expected, actual)
	}
}

func TestMatDenseSlice(t *testing.T) {
	var actual string
	actual = fmt.Sprintf("%g", mat.Formatted(MatDenseSlice(mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6}), 1, 2, 1, 2)))
	if actual != "[4]" {
		t.Errorf("got: %s", actual)
	}
	actual = fmt.Sprintf("%g", mat.Formatted(MatDenseRowSlice(mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), 1, 2)))
	if actual != "[3  4]" {
		t.Errorf("got: %s", actual)
	}
	actual = fmt.Sprintf("%g", mat.Formatted(MatDenseColSlice(mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), 1, 2).T()))
	if actual != "[2  4  6]" {
		t.Errorf("got: %s", actual)
	}
}

func TestToDense(t *testing.T) {
	m := MatConst{Rows: 1, Columns: 1, Value: 1}
	d := ToDense(m)
	if fmt.Sprintf("%g", mat.Formatted(d)) != "[1]" {
		t.Fail()
	}
	d2 := mat.NewDense(1, 1, []float64{0})
	m2 := FromDense(d2, d)
	if fmt.Sprintf("%g", mat.Formatted(m2)) != "[1]" {
		t.Fail()
	}

}
