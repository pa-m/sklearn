package linearmodel

import (
	"fmt"
	"github.com/pa-m/sklearn/metrics"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"testing"
	"time"
)

func TestBayesianRidge(t *testing.T) {
	nSamples, nFeatures, nOutputs := 10000, 5, 5
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
	for _, normalize := range []bool{false, true} {
		m := NewBayesianRidge()
		m.Normalize = normalize
		//m.Verbose = true
		m.ComputeScore = true
		start := time.Now()
		m.Fit(X, Y)
		elapsed := time.Since(start)
		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		m.Predict(X, Ypred)
		r2score := metrics.R2Score(Y, Ypred, nil, "variance_weighted").At(0, 0)
		unused(elapsed)
		//fmt.Printf("TestBayesianRidge normalize=%v score:%.4g elapsed:%s\n", m.Normalize, r2score, elapsed)
		if r2score < .99 {
			t.Fail()
		}
		// Output:
		// 75.
	}
}

func ExampleBayesianRidge() {
	nSamples, nFeatures, nOutputs := 10000, 5, 5
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
	m := NewBayesianRidge()
	//start := time.Now()
	m.Fit(X, Y)
	//elapsed := time.Since(start)
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	m.Predict(X, Ypred)
	r2score := metrics.R2Score(Y, Ypred, nil, "variance_weighted").At(0, 0)
	if r2score > .999 {
		fmt.Println("BayesianRidge ok")
	}
	// Output:
	// BayesianRidge ok

}
