package linearmodel

import (
	"fmt"
	"golang.org/x/exp/rand"
	"math"
	"testing"
	"time"

	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

func TestRidge(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	randmat := func(m, n int) *mat.Dense {
		M := mat.NewDense(m, n, nil)
		M.Apply(func(_, _ int, _ float64) float64 { return rand.Float64() }, M)
		return M
	}
	X := randmat(nSamples, nOutputs)
	W := randmat(nFeatures, nOutputs)
	Y := &mat.Dense{}
	Y.Mul(X, W)

	for _, normalize := range []bool{false} {

		regr := NewRidge()
		regr.Alpha = 0.
		regr.Tol = 1e-2
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(X, Y)
		elapsed := time.Since(start)
		unused(elapsed)
		//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		regr.Predict(X, Ypred)
		r2score := metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
		mse := metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
		mae := metrics.MeanAbsoluteError(Y, Ypred, nil, "").At(0, 0)
		if math.Sqrt(mse) > regr.Tol {
			t.Errorf("Test %T normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, normalize, r2score, metrics.R2Score(Y, Ypred, nil, "raw_values"), mse, mae)
			t.Fail()
		}
	}

}

func ExampleRidge() {
	X, Y := mat.NewDense(3, 2, []float64{0, 0, 1, 1, 2, 2}), mat.NewDense(3, 2, []float64{0, 0, 1, 1, 2, 2})
	clf := NewRidge()
	clf.Tol = 1e-3
	clf.Normalize = false
	clf.Alpha = 1
	clf.L1Ratio = 0.
	clf.Fit(X, Y)
	fmt.Printf("Coef:\n%.2f\n", mat.Formatted(clf.Coef.T()))
	fmt.Printf("Intercept:\n%.2f\n", mat.Formatted(clf.Intercept.T()))
	Ypred := &mat.Dense{}
	clf.Predict(X, Ypred)
	fmt.Printf("Ypred:\n%.2f\n", mat.Formatted(Ypred))

	// Output:
	// Coef:
	// ⎡0.40  0.40⎤
	// ⎣0.40  0.40⎦
	// Intercept:
	// ⎡0.20⎤
	// ⎣0.20⎦
	// Ypred:
	// ⎡0.20  0.20⎤
	// ⎢1.00  1.00⎥
	// ⎣1.80  1.80⎦

}
