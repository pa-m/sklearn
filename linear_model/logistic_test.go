package linearModel

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestLogReg(t *testing.T) {

	X, Ytrue := datasets.LoadExamScore()
	nSamples, nFeatures := X.Dims()

	preprocessing.InsertOnes(X)
	fmt.Println("nFeatures", nFeatures)
	nFeatures++
	//fmt.Println(base.MatStr(X.Slice(0, 10, 0, nFeatures)))

	_, nOutputs := Ytrue.Dims()
	regr := NewLogisticRegression()
	regr.FitIntercept = false // Fintintercept=false because we already added ones columns instead
	//regr.Normalize = false
	/*regr.Optimizer = func() base.Optimizer {
		s := base.NewAdamOptimizer()
		s.StepSize = .5
		return s
	}()*/
	regr.Coef = mat.NewDense(nFeatures, nOutputs, nil)
	regr.Alpha = 0. // following test values require no regularization
	//regr.Options.Epochs = int(4e6 / float64(nSamples))
	//regr.Options.GOMethod = &optimize.LBFGS{}

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)
	grad := mat.NewDense(nFeatures, nOutputs, nil)

	J := math.Inf(1)
	loss := func() float64 {
		return regr.LossFunction(Ytrue, X, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction)
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := []float64{grad.At(0, 0), grad.At(1, 0), grad.At(2, 0)}
		//fmt.Printf("%s grad=%v expected %v\n", context, actualGradient, expectedGradient)
		for j := 0; j < nFeatures; j++ {
			if !floats.EqualWithinAbs(expectedGradient[j], actualGradient[j], 1e-4) {
				t.Errorf("%s grad=%v expected %v", context, actualGradient, expectedGradient)
				return
			}
		}
	}
	chkTheta := func(context string, expectedTheta []float64) {
		actualTheta := []float64{regr.Coef.At(0, 0), regr.Coef.At(1, 0), regr.Coef.At(2, 0)}
		for j := 0; j < nFeatures; j++ {
			if !floats.EqualWithinAbs(expectedTheta[j], actualTheta[j], 1e-2) {
				t.Errorf("%s theta=%v expected %v", context, actualTheta, expectedTheta)
				return
			}
		}
	}
	J = loss()
	chkLoss("initial loss", 0.693)
	chkGrad("initial gradient", []float64{-0.1000, -12.0092, -11.2628})

	regr.Coef.SetCol(0, []float64{-24, 0.2, 0.2})
	J = loss()
	chkLoss("at test theta", 0.218)

	regr.Fit(X, Ytrue)
	J = loss()

	chkLoss("after fit", 0.2030)
	chkTheta("after fit", []float64{-25.161, 0.206, 0.201})

	regr.PredictProba(X, Ypred)
	fmt.Println("mae", metrics.MeanAbsoluteError(Ytrue, Ypred, nil, "uniform_average").At(0, 0))

	regr.PredictProba(X, Ypred)
	fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred, nil, "uniform_average").At(0, 0))
	fmt.Println("ok")
}

// Test differents normalize setup for LinearRegression
func _TestLogisticRegression(t *testing.T) {
	nSamples, nFeatures, nOutputs := 200, 2, 2
	p := NewRandomLinearProblem(nSamples, nFeatures, nOutputs)
	activation := Sigmoid{}
	p.Y.Apply(func(i int, o int, y float64) float64 {
		y = activation.F(y)
		if y >= .5 {
			return 1.
		}
		return 0.
	}, p.Y)

	bestErr := make(map[string]float)
	bestTime := time.Second * 86400
	bestSetup := make(map[string]string)

	for _, normalize := range []bool{false} {
		for _, optimizer := range []base.Optimizer{ /*base.NewSGDOptimizer(), base.NewAdagradOptimizer(), base.NewRMSPropOptimizer(), base.NewAdadeltaOptimizer(),*/ base.NewAdamOptimizer()} {
			testSetup := fmt.Sprintf("%s %v", optimizer, normalize)
			regr := NewLogisticRegression()
			regr.Normalize = normalize
			regr.Optimizer = optimizer
			regr.Alpha = 0.
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
			accuracy := metrics.AccuracyScore(p.Y, Ypred, nil, "").At(0, 0)
			fmt.Println("R2", metrics.R2Score(p.Y, Ypred, nil, "").At(0, 0))
			fmt.Println("MAE", metrics.MeanAbsoluteError(p.Y, Ypred, nil, "").At(0, 0))
			tmpScore, ok := bestErr["accuracy"]
			if !ok || accuracy > tmpScore {
				bestErr["accuracy"] = accuracy
				bestSetup["accuracy"] = testSetup + fmt.Sprintf("(%g)", accuracy)
			}
			if accuracy < .99 {
				t.Errorf("Test LogisticRegression %s normalize=%v accuracy=%g \n", optimizer, normalize, accuracy)
				t.Fail()
			} else {
				//fmt.Printf("Test LogisticRegression %s ok normalize=%v accuracy=%g elapsed=%s\n", optimizer, normalize, accuracy, elapsed)
			}
		}
	}
	fmt.Printf("Test Logisticregression BEST SETUP:%v\n\n", bestSetup)
}
