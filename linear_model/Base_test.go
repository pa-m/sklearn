package linearModel

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
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

func ExampleLinearRegression() {
	// adapted from http://scikit-learn.org/stable/_downloads/plot_ols.ipynb

	// # Load the diabetes dataset
	diabetes := datasets.LoadDiabetes()

	// # Use only one feature
	NSamples, _ := diabetes.X.Dims()
	diabetesX := diabetes.X.Slice(0, NSamples, 2, 3).(*mat.Dense)

	// # Split the data into training/testing sets
	diabetesXtrain := diabetesX.Slice(0, NSamples-20, 0, 1).(*mat.Dense)
	diabetesXtest := diabetesX.Slice(NSamples-20, NSamples, 0, 1).(*mat.Dense)

	// # Split the targets into training/testing sets
	diabetesytrain := diabetes.Y.Slice(0, NSamples-20, 0, 1).(*mat.Dense)
	diabetesytest := diabetes.Y.Slice(NSamples-20, NSamples, 0, 1).(*mat.Dense)

	// # Create linear regression object
	regr := NewLinearRegression()

	// # Train the model using the training sets
	regr.Fit(diabetesXtrain, diabetesytrain)

	// # Make predictions using the testing set
	NTestSamples := 20
	diabetesypred := mat.NewDense(NTestSamples, 1, nil)
	regr.Predict(diabetesXtest, diabetesypred)

	// # The coefficients
	fmt.Printf("Coefficients: %.3f\n", mat.Formatted(regr.Coef))
	// # The mean squared error
	fmt.Printf("Mean squared error: %.2f\n", metrics.MeanSquaredError(diabetesytest, diabetesypred, nil, "").At(0, 0))
	// # Explained variance score: 1 is perfect prediction
	fmt.Printf("Variance score: %.2f\n", metrics.R2Score(diabetesytest, diabetesypred, nil, "").At(0, 0))

	//   # Plot outputs
	canPlot := false
	if canPlot {

		// plot result
		p, _ := plot.New()

		xys := func(X, Y mat.Matrix) plotter.XYs {
			var data plotter.XYs
			NTestSamples, _ = X.Dims()
			for sample := 0; sample < NTestSamples; sample++ {
				data = append(data, struct{ X, Y float64 }{X.At(sample, 0), Y.At(sample, 0)})
			}
			return data
		}
		s, _ := plotter.NewScatter(xys(diabetesXtest, diabetesytest))
		l, _ := plotter.NewLine(xys(diabetesXtest, diabetesypred))
		l.Color = color.RGBA{0, 0, 255, 255}
		p.Add(s, l)

		// Save the plot to a PNG file.
		pngfile := "/tmp/linearregression.png"
		os.Remove(pngfile)
		if err := p.Save(4*vg.Inch, 3*vg.Inch, pngfile); err != nil {
			panic(err)
		}
		cmd := exec.Command("display", pngfile)
		err := cmd.Start()
		if err != nil {
			fmt.Println(err.Error())
		}
		time.Sleep(200 * time.Millisecond)
		os.Remove(pngfile)
	}

	// Output:
	// Coefficients: [938.238]
	// Mean squared error: 2548.07
	// Variance score: 0.47
}

// Test differents normalize setup for LinearRegression
func TestLinearRegression(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)

	sgd := base.NewSGDOptimizer()
	sgd.StepSize = .05
	for _, normalize := range []bool{false} { //true

		for _, optimizer := range []base.Optimizer{
			sgd,
			//base.NewAdagradOptimizer(),
			//base.NewRMSPropOptimizer(),
			base.NewAdadeltaOptimizer(), base.NewAdamOptimizer()} {
			testSetup := fmt.Sprintf("%s %v", optimizer, normalize)
			regr := NewLinearRegression()
			regr.Alpha = 0.

			regr.Normalize = normalize
			regr.Optimizer = optimizer
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
			if math.Sqrt(mse) > regr.Tol {
				t.Errorf("Test %T %s normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, optimizer, normalize, r2score, metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
				t.Fail()
			} else {
				//fmt.Printf("Test %T %s ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, optimizer, normalize, r2score, mse, mae, elapsed)
			}
		}
	}
	fmt.Printf("Test %T BEST SETUP:%v\n\n", LinearRegression{}, bestSetup)
}

func TestRidge(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	for _, normalize := range []bool{false} {

		regr := NewRidge()
		regr.Alpha = 0.
		regr.Tol = 1e-2
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		unused(elapsed)
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
			//fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
		}
	}

}

func TestLasso(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	for _, normalize := range []bool{false} {

		regr := NewLasso()
		regr.Alpha = .1
		regr.Tol = 1e-3
		regr.Normalize = normalize
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		unused(elapsed)
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
			//fmt.Printf("Test %T ok normalize=%v r2score=%g  mse=%g mae=%g elapsed=%s\n", regr, normalize, r2score, mse, mae, elapsed)
		}
	}

}

// ----

// TestSGDRegressor tests differents Method/Normalize setups for SGDRegressor
func TestGonumOptimizeRegressor(t *testing.T) {
	nSamples, nFeatures, nOutputs := 100, 5, 4
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
				//fmt.Printf("Test %T %12T ok normalize=%v\nr2score=%g  mse=%g mae=%g elapsed=%s\n", regr, method, normalize, r2score, mse, mae, elapsed)
			}
		}
	}
	// Best setup is usually Method:&optimize.LBFGS,Normalize:false
	fmt.Printf("Test %T BEST SETUP:%v\n\n", SGDRegressor{}, bestSetup)

}

func TestLinFitGOM(t *testing.T) {
	nSamples, nFeatures, nOutputs := 100, 5, 4
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)
	for _, methodCreator := range []func() optimize.Method{
		func() optimize.Method { return &optimize.GradientDescent{} },
		func() optimize.Method { return &optimize.BFGS{} },
		func() optimize.Method { return &optimize.CG{} },
		func() optimize.Method { return &optimize.LBFGS{} },
	} {
		for _, normalize := range []bool{false, true} {
			testSetup := fmt.Sprintf("%T %v", methodCreator(), normalize)
			//fmt.Printf("-- TestLinearRegression normalize=%v --\n", normalize)
			regr := NewRidge()
			regr.Alpha = 0.
			regr.Normalize = normalize
			regr.Options.GOMethodCreator = methodCreator
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
				t.Errorf("Test %T %12T normalize=%v\nr2score=%g (%v) mse=%g mae=%g \n", regr, methodCreator(), normalize, r2score, *metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
				t.Fail()
			} else {
				//fmt.Printf("Test %T %12T ok normalize=%v\nr2score=%g  mse=%g mae=%g elapsed=%s\n", regr, method, normalize, r2score, mse, mae, elapsed)
			}
		}
	}
	// Best setup is usually Method:&optimize.LBFGS,Normalize:false
	fmt.Printf("Test %T BEST SETUP:%v\n\n", SGDRegressor{}, bestSetup)

}

// TestBestRegressionImplementation test between base.Optimizer/BayesianRidge
func TestBestRegressionImplementation(t *testing.T) {
	nSamples, nFeatures, nOutputs := 100, 5, 5
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)
	for _, regr := range []base.Regressor{NewLinearRegression(), NewSGDRegressor(), NewBayesianRidge(), NewLasso()} {
		//for _, normalize := range []bool{false, true} {
		testSetup := fmt.Sprintf("%T", regr)

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
		if r2score < .95 {
			t.Errorf("Test %s\nr2score=%g (%v) mse=%g mae=%g \n", testSetup, r2score, *metrics.R2Score(p.Y, Ypred, nil, "raw_values"), mse, mae)
			t.Fail()
		} else {
			//fmt.Printf("Test %s ok\nr2score=%g  mse=%g mae=%g elapsed=%s\n", testSetup, r2score, mse, mae, elapsed)
		}
		//}
	}
	// Best setup is usually Method:&optimize.LBFGS,Normalize:false
	fmt.Printf("Test Regression implementations BEST SETUP:%v\n\n", bestSetup)

}
