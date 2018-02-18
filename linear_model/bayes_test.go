package linear_model

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestBayesianRidge(t *testing.T) {

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
	eps := .1
	Xp := [][]float{[]float{7., 8., 9.}}
	y_true := []float{f(Xp[0])}
	Yp := m.Predict(Xp)
	fmt.Println(Yp[0], " expected: ", y_true)
	if math.Abs(Yp[0]-y_true[0]) > eps {
		fmt.Printf("Yp[0]-y_true[0]=%g\n", Yp[0]-y_true[0])
		t.Fail()
	}
	// Output:
	// 75.
}

func ExampleBayesianRidge() {
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
	//fmt.Printf("Scores: %#v\n", m.Scores_)
	Xp := [][]float{[]float{7., 8., 9.}}
	y_true := []float{f(Xp[0])}
	Yp := m.Predict(Xp)
	fmt.Println(Yp[0], " expected: ", y_true)
	fmt.Printf("TestBayesianRidge score:%g\n", m.Score(X, Y, nil))
	// Output:
	// Convergence after  1  iterations
	// 74.952345026144  expected:  [75]
	// TestBayesianRidge score:0.991576356387405

}
