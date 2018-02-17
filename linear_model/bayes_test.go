package linear_model

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestBayesianRidge(t *testing.T) {

	fmt.Println("Hello")
	var X [][]float = make([][]float, 10000)
	Y := make([]float, len(X))
	f := func(X []float) float { return 1. + 2.*X[0] + 3.*X[1] + 4.*X[2] }
	for i := range X {
		X[i] = make([]float, 3, 3)
		for j := range X[i] {
			X[i][j] = rand.Float64()
		}
		Y[i] = f(X[i]) + (rand.Float64()-.5)/2
	}
	m := NewBayesianRidge()
	m.Normalize = true
	m.Verbose = true
	m.ComputeScore = true
	m.Fit(X, Y)
	fmt.Printf("Scores: %#v\n", m.Scores_)
	eps := .1
	Xp := [][]float{[]float{7., 8., 9.}}
	Yp := m.Predict(Xp)
	fmt.Println(Yp[0], " expected: ", f(Xp[0]))
	if math.Abs(Yp[0]-f(Xp[0])) > eps {
		fmt.Println(Yp[0], " au lieu de ", f(Xp[0]))
		t.Fail()
	}
	// Output:
	// 75.
}
