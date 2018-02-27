package linearModel

import (
	"fmt"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
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

func TestBayesianRidgeMat(t *testing.T) {
	nSamples, nFeatures, nOutputs := 10000, 3, 5
	X := mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i int, j int, v float64) float64 {
		return rand.NormFloat64() * 20
	}, X)
	f := func(X mat.Matrix, i, o int) float {
		if o == 0 {
			return 1. + 2.*X.At(i, 0) + 3.*X.At(i, 1) + 4.*X.At(i, 2)
		}
		return 1. - 2.*X.At(i, 0) + 3.*X.At(i, 1) + float64(o)*X.At(i, 2)

	}
	Y := mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i int, o int, v float64) float64 {
		return f(X, i, o)
	}, Y)

	m := NewBayesianRidgeMat()
	m.Normalize = true
	//m.Verbose = true
	m.ComputeScore = true
	start := time.Now()
	m.Fit(X, Y)
	elapsed := time.Since(start)
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	m.Predict(X, Ypred)
	r2score := metrics.R2Score(Y, Ypred, nil, "variance_weighted").At(0, 0)
	fmt.Printf("TestBayesianRidgeMat normalize=%v score:%.4g elapsed:%s\n", m.Normalize, r2score, elapsed)
	if r2score < .99 {
		t.Fail()
	}
	// Output:
	// 75.

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
