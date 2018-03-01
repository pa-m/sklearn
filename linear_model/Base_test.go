package linearModel

import (
	"fmt"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"math"
	"math/rand"
	"testing"
	"time"
)

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func NewRandomLinearProblem(nSamples, nFeatures, nOutputs int) *Problem {
	X := mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, v float64) float64 {
		if j == 0 {
			return 1.
		}
		return rand.NormFloat64() * 10.
	}, X)
	TrueTheta := mat.NewDense(nFeatures, nOutputs, nil)
	TrueTheta.Apply(func(i, j int, v float64) float64 {
		return rand.NormFloat64() * 10.
	}, TrueTheta)
	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
	Ytrue.Product(X, TrueTheta)
	return &Problem{X: X, Y: Ytrue}
}

func TestLinearRegression(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	for _, normalize := range []bool{false, true} {

		regr := NewLinearRegression()
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		regr.Predict(p.X, Ypred)
		r2score := metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0)
		mse := metrics.MeanSquaredError(p.Y, Ypred, nil, "").At(0, 0)
		mae := metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0)
		if math.Sqrt(mse) > regr.Tol {
			t.Errorf("Test %T normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, normalize, r2score, metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
			t.Fail()
		} else {
			fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
		}
	}
}

func TestRidge(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	for _, normalize := range []bool{false, true} {

		regr := NewRidge()
		regr.Alpha = .1
		regr.Tol = 1e-2
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		regr.Predict(p.X, Ypred)
		r2score := metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0)
		mse := metrics.MeanSquaredError(p.Y, Ypred, nil, "").At(0, 0)
		mae := metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0)
		if math.Sqrt(mse) > regr.Tol {
			t.Errorf("Test %T normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, normalize, r2score, metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
			t.Fail()
		} else {
			fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
		}
	}

}

func TestLasso(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	for _, normalize := range []bool{false, true} {

		regr := NewLasso()
		regr.Alpha = .1
		regr.Tol = 1e-3
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
		Ypred := mat.NewDense(nSamples, nOutputs, nil)
		regr.Predict(p.X, Ypred)
		r2score := metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0)
		mse := metrics.MeanSquaredError(p.Y, Ypred, nil, "").At(0, 0)
		mae := metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0)
		if math.Sqrt(mse) > regr.Tol {
			t.Errorf("Test %T normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, normalize, r2score, metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
			t.Fail()
		} else {
			fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
		}
	}

}

// ----
func TestSGDRegressor(t *testing.T) {
	nFeatures, nSamples, nOutputs := 100, 5, 5
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)
	for _, method := range []optimize.Method{&optimize.GradientDescent{}, &optimize.BFGS{}, &optimize.CG{}, &optimize.LBFGS{}} {
		for _, normalize := range []bool{false, true} {
			testSetup := fmt.Sprintf("%T %v", method, normalize)
			//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
			regr := NewSGDRegressor()
			regr.Alpha = 0.
			regr.Method = method
			regr.Normalize = normalize
			//m.Verbose = true
			//m.ComputeScore = true
			start := time.Now()
			regr.Fit(p.X, p.Y)
			elapsed := time.Since(start)
			//fmt.Println("XOffset", regr.XOffset, "Intercept", regr.Intercept, "Coef", regr.Coef)
			Ypred := mat.NewDense(nSamples, nOutputs, nil)
			regr.Predict(p.X, Ypred)

			if elapsed < bestTime {
				bestTime = elapsed
				bestSetup["elapsed"] = testSetup + fmt.Sprintf("(%s)", elapsed)
			}
			r2score := metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0)
			tmpScore, ok := bestErr["R2"]
			if !ok || r2score > tmpScore {
				bestErr["R2"] = r2score
				bestSetup["R2"] = testSetup + fmt.Sprintf("(%g)", r2score)
			}
			mse := metrics.MeanSquaredError(p.Y, Ypred, nil, "").At(0, 0)
			tmpScore, ok = bestErr["MSE"]
			if !ok || mse < tmpScore {
				bestErr["MSE"] = mse
				bestSetup["MSE"] = testSetup + fmt.Sprintf("(%g)", mse)
			}
			mae := metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0)
			tmpScore, ok = bestErr["MAE"]
			if !ok || mae < tmpScore {
				bestErr["MAE"] = mae
				bestSetup["MAE"] = testSetup + fmt.Sprintf("(%g)", mae)
			}
			//if math.Sqrt(mse) > regr.Tol {
			if r2score < .99 {
				t.Errorf("Test %T %12T normalize=%v\nr2score=%g (%v) mse=%g mae=%g \n", regr, method, normalize, r2score, *metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
				t.Fail()
			} else {
				fmt.Printf("Test %T %12T ok normalize=%v\nr2score=%g  mse=%g mae=%g elapsed=%s\n", regr, method, normalize, r2score, mse, mae, elapsed)
			}
		}
	}
	// Best setup is usually Method:&optimize.LBFGS,Normalize:false
	fmt.Printf("Test %T BEST SETUP:%v\n\n", SGDRegressor{}, bestSetup)

}
