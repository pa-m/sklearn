package linearModel

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestBayesianRidge(t *testing.T) {

	X := make([][]float, 10000)
	Y := make([]float, len(X))
	f := func(X []float) float { return 1. + 2.*X[0] + 3.*X[1] + 4.*X[2] }
	for i := range X {
		X[i] = make([]float, 3, 3)
		for j := range X[i] {
			X[i][j] = rand.Float64()*20. - 10.
		}
		Y[i] = f(X[i]) + (rand.Float64()-.5)/2
	}
	for _, normalize := range []bool{false, true} {
		m := NewBayesianRidge()
		m.Normalize = normalize
		//m.Verbose = true
		m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		fmt.Printf("TestBayesianRidge normalize=%v score:%.4g elapsed:%s\n", normalize, m.Score(X, Y, nil), elapsed)
		eps := .1
		Xp := [][]float{{7., 8., 9.}}
		yTrue := []float{f(Xp[0])}
		Yp := m.Predict(Xp)
		//fmt.Println(Yp[0], " expected: ", yTrue)
		if math.Abs(Yp[0]-yTrue[0]) > eps {
			fmt.Printf("TestBayesianRidge Yp[0]-yTrue[0]=%g\n", Yp[0]-yTrue[0])
			t.Fail()
		}
		// Output:
		// 75.
	}
}

func ExampleBayesianRidge() {
	X := make([][]float, 10000)
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
	//m.Verbose = true
	m.ComputeScore = true
	m.Fit(X, Y)
	//fmt.Printf("Scores: %#v\n", m.Scores_)
	Xp := [][]float{{7., 8., 9.}}
	yTrue := []float{f(Xp[0])}
	Yp := m.Predict(Xp)
	unused(yTrue, Yp)

	//fmt.Println(Yp[0], " expected: ", yTrue)
	fmt.Printf("TestBayesianRidge score:%.2g\n", m.Score(X, Y, nil))
	// Output:
	// TestBayesianRidge score:0.99
}
