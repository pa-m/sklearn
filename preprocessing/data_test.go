package preprocessing

import (
	"fmt"
	"github.com/gonum/floats"
	_ "math"

	"testing"
)

func TestMinMaxScaler(t *testing.T) {
	m := NewMinMaxScaler([]float{0, 1})
	m.Fit([][]float{
		{1, 2, 3},
		{1, 4, 7},
		{1, 5, 9},
	}, nil)
	if !floats.EqualApprox(m.Scale, []float{1, 1. / 3, 1. / 6}, 1e-6) {
		fmt.Println("bad scale")
		t.Fail()
	}
	m.Fit([][]float{
		{1, 2, 3},
		{1, 4, 7},
		{9, 5, 9},
	}, nil)
	X := [][]float{{1, 2, 3}}
	Y := m.Transform(X)
	if !floats.EqualApprox(Y[0], []float{0, 0, 0}, 1e-6) {
		fmt.Println("bad min")
		t.Fail()
	}
	X = [][]float{{9, 5, 9}}
	Y = m.Transform(X)
	if !floats.EqualApprox(Y[0], []float{1, 1, 1}, 1e-6) {
		fmt.Printf("bad Y=%v\n", Y)
		t.Fail()
	}

	m = NewMinMaxScaler([]float{0, 10})
	m.Fit([][]float{
		{1, 2, 3},
		{1, 4, 7},
		{9, 5, 9},
	}, nil)
	X = [][]float{{8, 8, 8}}
	Y = m.Transform(X)
	X2 := m.InverseTransform(Y)
	if !floats.EqualApprox(X[0], X2[0], 1e-6) {
		t.Fail()
	}
}

func TestStandardScaler(t *testing.T) {
	m := NewStandardScaler()
	m.Fit([][]float{
		{1, 2, 3},
		{1, 4, 7},
		{9, 5, 9},
	}, nil)
	X := [][]float{{8, 8, 8}}
	Y := m.Transform(X)
	//fmt.Printf("Y=%#v\n", Y)
	X2 := m.InverseTransform(Y)
	if !floats.EqualApprox(X[0], X2[0], 1e-6) {
		t.Fail()
	}
}

func TestPolynomialFeatures(t *testing.T) {
	pf := NewPolynomialFeatures()
	pf.Degree = 3
	fmt.Println("TestPolynomialFeatures")
	pf.IncludeBias = true
	pf.InteractionOnly = false
	X := [][]float{{1, 2, 3}}
	pf.Fit(X, nil)
	fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Fail()
	}
	fmt.Printf("poly X:%v\n", pf.Transform(X))

	pf.IncludeBias = true
	pf.InteractionOnly = true
	pf.Fit([][]float{{1, 2, 3}}, nil)
	fmt.Printf("powers interactiononly=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 0] [0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 2 0] [0 3 0] [1 0 0] [2 0 0] [3 0 0]]" {
		fmt.Println("failed interactiononly")
		t.Fail()
	}

	pf.IncludeBias = false
	pf.InteractionOnly = false
	pf.Fit([][]float{{1, 2, 3}}, nil)
	fmt.Printf("powers=%v\n", pf.Powers)
	if fmt.Sprintf("%v", pf.Powers) != "[[0 0 1] [0 0 2] [0 0 3] [0 1 0] [0 1 1] [0 1 2] [0 2 0] [0 2 1] [0 3 0] [1 0 0] [1 0 1] [1 0 2] [1 1 0] [1 1 1] [1 2 0] [2 0 0] [2 0 1] [2 1 0] [3 0 0]]" {
		t.Fail()
	}

}
