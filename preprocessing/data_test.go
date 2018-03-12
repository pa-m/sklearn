package preprocessing

import (
	"fmt"
	_ "math"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"testing"
)

func TestMinMaxScaler(t *testing.T) {
	m := NewMinMaxScaler([]float{0, 1})
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 1, 5, 9})
	m.Fit(X, nil)
	if !floats.EqualApprox(m.Scale.RawRowView(0), []float{1, 1. / 3, 1. / 6}, 1e-6) {
		t.Error("bad scale")
		t.Fail()
	}
	X = mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{1, 2, 3})
	Y, _ := m.Transform(X, nil)
	if !floats.EqualApprox(Y.RawRowView(0), []float{0, 0, 0}, 1e-6) {
		t.Error("bad min")
		t.Fail()
	}

	X = mat.NewDense(1, 3, []float64{9, 5, 9})
	Y, _ = m.Transform(X, nil)
	if !floats.EqualApprox(Y.RawRowView(0), []float{1, 1, 1}, 1e-6) {
		t.Errorf("bad Y=%v\n", Y)
		t.Fail()
	}

	m = NewMinMaxScaler([]float{0, 10})
	X = mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{8, 8, 8})
	Y, _ = m.Transform(X, nil)
	X2, _ := m.InverseTransform(Y, nil)
	if !floats.EqualApprox(X.RawRowView(0), X2.RawRowView(0), 1e-6) {
		t.Errorf("MinMaxScaler InverseTransform failed %v", X2.RawRowView(0))
		t.Fail()
	}
}

func TestStandardScaler(t *testing.T) {

	m := NewStandardScaler()
	isTransformer := func(Transformer) {}
	isTransformer(m)
	X := mat.NewDense(3, 3, []float64{1, 2, 3, 1, 4, 7, 9, 5, 9})
	m.Fit(X, nil)
	X = mat.NewDense(1, 3, []float64{8, 8, 8})
	Y, _ := m.Transform(X, nil)
	//fmt.Printf("Y=%#v\n", Y)
	X2, _ := m.InverseTransform(Y, nil)
	if !floats.EqualApprox(X.RawRowView(0), X2.RawRowView(0), 1e-6) {
		t.Errorf("StandardScaler inversetransform failed %v", X2.RawRowView(0))
		t.Fail()
	}
}

func TestPolynomialFeatures(t *testing.T) {
	pf := NewPolynomialFeatures(3)
	isTransformer := func(Transformer) {}
	isTransformer(pf)
	pf.IncludeBias = true
	pf.InteractionOnly = false
	X := mat.NewDense(1, 3, []float{1, 2, 3})
	pf.Fit(X, nil)
	//fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Fail()
	}
	Xout, _ := pf.Transform(X, nil)
	if "[1 3 9 27 2 6 18 4 12 8 1 3 9 2 6 4 1 3 2 1]" != fmt.Sprint(Xout.RawRowView(0)) {
		t.Fail()
	}

	pf.IncludeBias = true
	pf.InteractionOnly = true
	pf.Fit(X, nil)
	//fmt.Printf("powers interactiononly=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 2 0] [0 3 0] [1 0 0] [2 0 0] [3 0 0]]" {
		fmt.Println("failed interactiononly")
		t.Fail()
	}

	pf.IncludeBias = false
	pf.InteractionOnly = false
	pf.Fit(X, nil)
	//fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Fail()
	}

}

func ExampleInsertOnes() {
	X := mat.NewDense(2, 5, []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	InsertOnes(X)
	fmt.Printf("X %v\n", X.RawRowView(0))
	fmt.Printf("X %v\n", X.RawRowView(1))
	// Output:
	// X [1 2 3 4 5 6]
	// X [1 7 8 9 10 11]
}

func TestOneHotEncoder(t *testing.T) {
	Y := mat.NewDense(2, 1, []float64{1, 10})
	ohe := NewOneHotEncoder()
	ohe.Fit(nil, Y)
	//fmt.Printf("%#v\n", ohe)
	if ohe.Min[0] != 1 {
		t.Fail()
	}
	if ohe.NumClasses[0] != 10 {
		t.Fail()
	}
}
