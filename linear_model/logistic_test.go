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
	"gonum.org/v1/gonum/optimize"
)

func TestLogRegExamScore(t *testing.T) {
	X, Ytrue := datasets.LoadExamScore()
	nSamples, nFeatures := X.Dims()
	preprocessing.InsertOnes(X)
	nFeatures++
	_, nOutputs := Ytrue.Dims()
	regr := NewLogisticRegression()
	regr.FitIntercept = false // Fintintercept=false because we already added ones columns instead

	// we allocate Coef here because we use it for loss and grad tests before Fit
	regr.Coef = mat.NewDense(nFeatures, nOutputs, nil)

	// Alpha=0 here because following test values require no regularization
	regr.Alpha = 0.

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)
	grad := mat.NewDense(nFeatures, nOutputs, nil)

	J := math.Inf(1)
	loss := func() (J float64) {
		J = regr.LossFunction(Ytrue, X, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction)
		return
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

	// test Fit with various gonum/optimize Method

	var GOMethods = []optimize.Method{
		&optimize.BFGS{},
		&optimize.CG{},
		//&optimize.GradientDescent{},
		&optimize.LBFGS{},
		//&optimize.NelderMead{},
		//&optimize.Newton{}
	}
	printer := optimize.NewPrinter()
	//printer.HeadingInterval = 1
	//printer.ValueInterval = 0
	unused(printer)

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	for _, method := range GOMethods {
		testSetup := fmt.Sprintf("(%T)", method)
		regr.Coef.SetCol(0, []float64{-24, 0.2, 0.2})
		regr.Options.GOMethod = method
		//regr.Options.Recorder = printer
		start := time.Now()
		regr.Fit(X, Ytrue)
		elapsed := time.Since(start)
		J = loss()

		chkLoss("after fit "+testSetup, 0.2030)
		chkTheta("after fit "+testSetup, []float64{-25.161, 0.206, 0.201})
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
	}

	// test Fit with various base.Optimizer
	var Optimizers = []string{
		// "sgd",
		// "adagrad",
		// "rmsprop",
		//"adadelta",
		//"adam",
	}

	newOptimizer := func(name string) base.Optimizer {

		switch name {
		case "adadelta":
			s := base.NewAdadeltaOptimizer()
			s.StepSize = 0.05
			return s
		case "adam":
			s := base.NewAdamOptimizer()
			s.StepSize = .1
			return s
		default:
			s := base.NewOptimizer(name)
			return s
		}
	}

	for _, optimizer := range Optimizers {
		testSetup := optimizer
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.SetCol(0, []float64{-24, 0.2, 0.2})
		}
		regr.Options.GOMethod = nil
		//regr.Options.Recorder = printer
		regr.Options.MiniBatchSize = nSamples
		regr.Optimizer = newOptimizer(optimizer)
		// if opti, ok := regr.Optimizer.(*base.SGDOptimizer); ok {
		// 	opti.StepSize = 1e-5
		// }
		//regr.Options.Epochs = 1e5
		start := time.Now()
		regr.Fit(X, Ytrue)
		elapsed := time.Since(start)
		J = loss()

		chkLoss("after fit "+testSetup, 0.2030)
		chkTheta("after fit "+testSetup, []float64{-25.161, 0.206, 0.201})
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
	}
	fmt.Println("LogisticRegression BEST SETUP:", best)

	regr.PredictProba(X, Ypred)

	regr.PredictProba(X, Ypred)
	// fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred, nil, "uniform_average").At(0, 0))
	// fmt.Println("ok")
}

func TestLogRegMicrochipTest(t *testing.T) {
	X, Ytrue := datasets.LoadMicroChipTest()
	nSamples, nFeatures := X.Dims()

	//Xp, _ := preprocessing.NewPolynomialFeatures(6).Fit(X, Ytrue).Transform(X, Ytrue)
	// add poly features manually to have same order
	Xp := mat.NewDense(nSamples, 28, nil)
	c := 0
	for i := 0; i <= 6; i++ {
		for j := 0; j <= i; j++ {
			for s := 0; s < nSamples; s++ {
				Xp.Set(s, c, math.Pow(X.At(s, 0), float64(i-j))*math.Pow(X.At(s, 1), float64(j)))
			}
			c++
		}
	}

	_, nFeatures = Xp.Dims()
	_, nOutputs := Ytrue.Dims()
	regr := NewLogisticRegression()
	regr.FitIntercept = false // Fintintercept=false because we added polynomial features

	// we allocate Coef here because we use it for loss and grad tests before Fit
	regr.Coef = mat.NewDense(nFeatures, nOutputs, nil)

	// Alpha=1 here because following test values require no regularization
	regr.Alpha = 1.
	regr.L1Ratio = 0.

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	chkdims(".", Ypred, Xp, regr.Coef)
	Ydiff := mat.NewDense(nSamples, nOutputs, nil)
	grad := mat.NewDense(nFeatures, nOutputs, nil)

	J := math.Inf(1)
	loss := func() (J float64) {
		J = regr.LossFunction(Ytrue, Xp, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction)
		return
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := grad.RawRowView(0)[0:len(expectedGradient)]

		//fmt.Printf("%s grad=%v expected %v\n", context, actualGradient, expectedGradient)
		for j := 0; j < len(expectedGradient); j++ {
			if !floats.EqualWithinAbs(expectedGradient[j], actualGradient[j], 1e-4) {
				t.Errorf("%s grad=%v expected %v", context, actualGradient, expectedGradient)
				return
			}
		}
	}

	J = loss()
	chkLoss("Microchip initial loss", 0.693)
	chkGrad("Microchip initial gradient", []float64{0.0085, 0.0188, 0.0001, 0.0503, 0.0115})

	regr.Coef.Apply(func(j, o int, _ float64) float64 { return 1. }, regr.Coef)
	regr.Alpha = 10.

	J = regr.LossFunction(Ytrue, Xp, regr.Coef, Ypred, Ydiff, grad, regr.Alpha, regr.L1Ratio, nSamples, regr.ActivationFunction)
	chkLoss("At test theta", 3.164)
	chkGrad("at test theta", []float64{0.3460, 0.1614, 0.1948, 0.2269, 0.0922})

	// test Fit Microchip with various gonum/optimize Method

	var GOMethods = []optimize.Method{
		&optimize.BFGS{},
		&optimize.CG{},
		//&optimize.GradientDescent{},
		&optimize.LBFGS{},
		//&optimize.NelderMead{},
		//&optimize.Newton{}
	}
	printer := optimize.NewPrinter()
	//printer.HeadingInterval = 1
	//printer.ValueInterval = 0
	unused(printer)

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	for _, method := range GOMethods {
		testSetup := fmt.Sprintf("(%T)", method)
		regr.Options.GOMethod = method
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.Apply(func(j, o int, _ float64) float64 { return 0. }, Theta)
		}
		regr.Alpha = 1.

		//regr.Options.Recorder = printer
		start := time.Now()
		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()
		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, nil, "").At(0, 0)
		expectedAccuracy := 0.83
		if accuracy < expectedAccuracy {
			t.Errorf("%T accuracy=%g expected:%g", method, accuracy, expectedAccuracy)
		}
	}

	// test Fit with various base.Optimizer
	var Optimizers = []string{
		"sgd",
		"adagrad",
		//"rmsprop",
		"adadelta",
		"adam",
	}

	newOptimizer := func(name string) base.Optimizer {

		switch name {
		case "sgd":
			s := base.NewSGDOptimizer()
			s.StepSize = 0.95
			s.Momentum = .98
			return s
		case "rmsprop":
			s := base.NewRMSPropOptimizer()
			s.StepSize = .95
			s.RMSPropGamma = .5
			return s
		case "adadelta":
			s := base.NewAdadeltaOptimizer()
			s.StepSize = 0.05
			return s
		case "adam":
			s := base.NewAdamOptimizer()
			s.StepSize = .1
			return s
		default:
			s := base.NewOptimizer(name)
			return s
		}
	}

	for _, optimizer := range Optimizers {
		testSetup := optimizer
		regr.Options.ThetaInitializer = func(Theta *mat.Dense) {
			Theta.Apply(func(j, o int, _ float64) float64 { return 0. }, Theta)
		}
		regr.Options.GOMethod = nil
		//regr.Options.Recorder = printer
		regr.Options.MiniBatchSize = nSamples
		regr.Optimizer = newOptimizer(optimizer)
		// if opti, ok := regr.Optimizer.(*base.SGDOptimizer); ok {
		// 	opti.StepSize = 1e-5
		// }
		//regr.Options.Epochs = 1e5
		start := time.Now()
		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()

		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, nil, "").At(0, 0)
		expectedAccuracy := 0.80
		if accuracy < expectedAccuracy {
			t.Errorf("%s accuracy=%g expected:%g", regr.Optimizer, accuracy, expectedAccuracy)
		}
	}
	fmt.Println("LogisticRegression BEST SETUP:", best)

	// fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred, nil, "uniform_average").At(0, 0))
	// fmt.Println("ok")
}
