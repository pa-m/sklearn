package linear_model

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestLinearRegression(t *testing.T) {
	var X [][]float = make([][]float, 10000)
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
		//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
		m := NewLinearRegression()
		m.Normalize = normalize
		//m.Verbose = true
		//m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		fmt.Printf("TestLinearRegression normalize=%v score:%.4g elapsed:%s\n", normalize, m.Score(X, Y, nil), elapsed)
		eps := .1
		Xp := [][]float{{7., 8., 9.}}
		y_true := []float{f(Xp[0])}
		Yp := m.Predict(Xp)
		//fmt.Println(Yp[0], " expected: ", y_true)
		if math.Abs(Yp[0]-y_true[0]) > eps {
			fmt.Printf("TestLinearRegression Yp[0]-y_true[0]=%g\n", Yp[0]-y_true[0])
			t.Fail()
		}
		//fmt.Println("-----------------------------------------------------------")
		// Output:
		// 75.
	}

}

func TestRidge(t *testing.T) {
	var X [][]float = make([][]float, 10000)
	Y := make([]float, len(X))
	f := func(X []float) float { return 1. + 2.*X[0] + 3.*X[1] + 4.*X[2] }
	for i := range X {
		X[i] = make([]float, 3, 3)
		for j := range X[i] {
			X[i][j] = rand.Float64()*20. - 10.
		}
		Y[i] = f(X[i]) //+ (rand.Float64()-.5)/2
	}
	for _, normalize := range []bool{false, true} {
		//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
		m := NewRidge()
		m.Alpha = .1
		m.Normalize = normalize
		//m.Verbose = true
		//m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		score := m.Score(X, Y, nil)
		fmt.Printf("TestRidge normalize=%v score:%.4g elapsed:%s\n", normalize, score, elapsed)
		eps := .1
		Xp := [][]float{{7., 8., 9.}}
		y_true := []float{f(Xp[0])}
		Yp := m.Predict(Xp)
		//fmt.Println(Yp[0], " expected: ", y_true)
		if math.Abs(Yp[0]-y_true[0]) > eps {
			fmt.Printf("TestRidge normalize=%v Yp[0]-y_true[0]=%g\n", normalize, Yp[0]-y_true[0])
			t.Fail()
		}
		//fmt.Println("-----------------------------------------------------------")
		// Output:
		// 75.
	}

}

func TestLasso(t *testing.T) {
	var X [][]float = make([][]float, 10000)
	Y := make([]float, len(X))
	f := func(X []float) float { return 1. + 2.*X[0] + 3.*X[1] + 4.*X[2] }
	for i := range X {
		X[i] = make([]float, 3, 3)
		for j := range X[i] {
			X[i][j] = rand.Float64()*20. - 10.
		}
		Y[i] = f(X[i]) //+ (rand.Float64()-.5)/2
	}
	for _, normalize := range []bool{false, true} {
		//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
		m := NewLasso()
		m.Alpha = .1
		m.Normalize = normalize
		//m.Verbose = true
		//m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		score := m.Score(X, Y, nil)
		fmt.Printf("TestLasso normalize=%v score:%.4g elapsed:%s\n", normalize, score, elapsed)
		eps := .1
		Xp := [][]float{{7., 8., 9.}}
		y_true := []float{f(Xp[0])}
		Yp := m.Predict(Xp)
		//fmt.Println(Yp[0], " expected: ", y_true)
		if math.Abs(Yp[0]-y_true[0]) > eps {
			fmt.Printf("TestLasso normalise=%v Yp[0]-y_true[0]=%g\n", normalize, Yp[0]-y_true[0])
			t.Fail()
		}
		//fmt.Println("-----------------------------------------------------------")
		// Output:
		// 75.
	}

}

// ----
func TestSGDRegressor(t *testing.T) {
	var X [][]float = make([][]float, 10000)
	Y := make([]float, len(X))
	f := func(X []float) float { return 1. + 2.*X[0] + 3.*X[1] + 4.*X[2] }
	for i := range X {
		X[i] = make([]float, 3, 3)
		for j := range X[i] {
			X[i][j] = rand.Float64()*20. - 10.
		}
		Y[i] = f(X[i]) //+ (rand.Float64()-.5)/2
	}
	for _, normalize := range []bool{false, true} {
		//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
		m := NewSGDRegressor()
		m.LearningRate = 1e-2
		m.Normalize = normalize
		//m.Verbose = true
		//m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		score := m.Score(X, Y, nil)
		fmt.Printf("TestSGDRegressor normalize=%v score:%.4g elapsed:%s\n", normalize, score, elapsed)
		eps := .1
		Xp := [][]float{{7., 8., 9.}}
		y_true := []float{f(Xp[0])}
		Yp := m.Predict(Xp)
		//fmt.Println(Yp[0], " expected: ", y_true)
		if math.Abs(Yp[0]-y_true[0]) > eps {
			fmt.Printf("TestSGDRegressor normalise=%v Yp[0]-y_true[0]=%g\n", normalize, Yp[0]-y_true[0])
			t.Fail()
		}
		//fmt.Println("-----------------------------------------------------------")
		// Output:
		// 75.
	}

}
