package linearModel

import (
	"fmt"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"

	//	"math"
	"testing"
	"time"
)

// LinearRegressionGorgonia is a linear regr with sevelal outputs
func TestLinearRegressionGorgonia(t *testing.T) {
	start := time.Now()
	nSamples, nFeatures, nOutputs := 100, 5, 5
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	{
		for _, normalize := range []bool{false, true} {
			regr := NewLinearRegressionGorgonia()
			regr.Normalize = normalize
			regr.Tol = 1e-3
			regr.Fit(p.X, p.Y)
			elapsed := time.Since(start)
			//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
			Ypred := mat.NewDense(nSamples, nOutputs, nil)
			regr.Predict(p.X, Ypred)
			r2score := metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0)
			mse := metrics.MeanSquaredError(p.Y, Ypred, nil, "").At(0, 0)
			mae := metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0)
			if r2score < .99 {
				t.Errorf("Test %T normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, normalize, r2score, metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
				t.Fail()
			} else {
				fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
			}
		}
	}
}
