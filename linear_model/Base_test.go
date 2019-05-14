package linearmodel

import (
	"fmt"
	"image/color"
	"math"
	"os"
	"os/exec"
	"testing"
	"time"

	"golang.org/x/exp/rand"

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
	diabetesYtrain := diabetes.Y.Slice(0, NSamples-20, 0, 1).(*mat.Dense)
	diabetesYtest := diabetes.Y.Slice(NSamples-20, NSamples, 0, 1).(*mat.Dense)

	// # Create linear regression object
	regr := NewLinearRegression()

	// # Train the model using the training sets
	regr.Fit(diabetesXtrain, diabetesYtrain)

	// # Make predictions using the testing set
	NTestSamples := 20
	diabetesYpred := mat.NewDense(NTestSamples, 1, nil)
	regr.Predict(diabetesXtest, diabetesYpred)

	// # The coefficients
	fmt.Printf("Coefficients: %.3f\n", mat.Formatted(regr.Coef))
	// # The mean squared error
	fmt.Printf("Mean squared error: %.2f\n", metrics.MeanSquaredError(diabetesYtest, diabetesYpred, nil, "").At(0, 0))
	// # Explained variance score: 1 is perfect prediction
	fmt.Printf("Variance score: %.2f\n", metrics.R2Score(diabetesYtest, diabetesYpred, nil, "").At(0, 0))

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
		s, _ := plotter.NewScatter(xys(diabetesXtest, diabetesYtest))
		l, _ := plotter.NewLine(xys(diabetesXtest, diabetesYpred))
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

// Test differents normalize setup for RegularizedRegression
func TestRegularizedRegression(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)

	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)

	for _, normalize := range []bool{false} { //true

		for _, solver := range []string{"sgd", "adadelta", "adam"} {
			testSetup := fmt.Sprintf("%s %v", solver, normalize)
			regr := &RegularizedRegression{}
			regr.FitIntercept = true
			//regr.Options.PerOutputFit = true
			regr.Alpha = 0.

			regr.Normalize = normalize
			regr.Solver = solver
			regr.SolverConfigure = func(optimizer base.Optimizer) {
				switch solver {
				case "sgd":
					optimizer.(*base.SGDOptimizer).StepSize = .05
				case "adadelta":
					optimizer.(*base.SGDOptimizer).StepSize = .1
				}
			}
			regr.Tol = 1e-4
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
				t.Errorf("Test %T %s normalize=%v r2score=%g (%v) mse=%g mae=%g \n", regr, solver, normalize, r2score, mat.Formatted(metrics.R2Score(p.Y, Ypred, nil, "raw_values")), mse, mae)
				t.Fail()
			}
		}
	}
	fmt.Printf("Test %T BEST SETUP:%v\n\n", LinearRegression{}, bestSetup)
}

// ----

// TestSGDRegressor tests differents Method/Normalize setups for SGDRegressor
func TestGonumOptimizeRegressor(t *testing.T) {
	nSamples, nFeatures, nOutputs := 100, 5, 4
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)
	for _, method := range []optimize.Method{ /*&optimize.GradientDescent{}, &optimize.BFGS{}, &optimize.CG{},*/ &optimize.LBFGS{}} {
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
			regr := &RegularizedRegression{}
			regr.FitIntercept = true
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
	for _, regr := range []base.Predicter{NewRidge(), NewLasso(), NewSGDRegressor()} {
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
		}
		//}
	}
	// Best setup is usually Method:&optimize.LBFGS,Normalize:false
	fmt.Printf("Test Regression implementations BEST SETUP:%v\n\n", bestSetup)

}

func _ExampleSetIntercept() {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 7, 3, 6})
	W := mat.NewDense(2, 2, []float64{5, 6, 7, 8})
	Y := &mat.Dense{}
	Y.Mul(X, W)
	Y.Add(Y, mat.NewDense(3, 2, []float64{10, 10, 10, 10, 10, 10}))
	lm := NewLinearRegression()
	lm.Fit(X, Y)
	fmt.Printf("%.2f\n", mat.Formatted(lm.Coef))
	fmt.Printf("%.2f\n", mat.Formatted(lm.Intercept))
	// Output:
	// ⎡5.00  6.00⎤
	// ⎣7.00  8.00⎦
	// [10.00  10.00]

}
