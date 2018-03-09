package neuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"pa-m/sklearn/datasets"
	"pa-m/sklearn/metrics"
	"testing"
	"time"

	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
	lm "github.com/pa-m/sklearn/linear_model"

	"gonum.org/v1/gonum/mat"
)

type Problem struct {
	X, Y          *mat.Dense
	MiniBatchSize int
}

func NewRandomProblem(nSamples, nFeatures, nOutputs int, activation string, loss string) *Problem {
	X := mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, X)
	TrueTheta := mat.NewDense(nFeatures, nOutputs, nil)
	TrueTheta.Apply(func(i, j int, v float64) float64 {
		return rand.Float64()
	}, TrueTheta)
	Z := mat.NewDense(nSamples, nOutputs, nil)
	Ytrue := mat.NewDense(nSamples, nOutputs, nil)
	Z.Mul(X, TrueTheta)
	NewActivation(activation).Func(Z, Ytrue)
	if loss == "cross-entropy" {
		for sample := 0; sample < nSamples; sample++ {
			oTrue := floats.MaxIdx(Ytrue.RawRowView(sample))
			for o := 0; o < nOutputs; o++ {
				y := 0.
				if o == oTrue {
					y = 1
				}
				Ytrue.Set(sample, o, y)
			}
		}
	}
	return &Problem{X: X, Y: Ytrue}
}

func TestMLPRegressorIdentitySquareLoss(t *testing.T) {
	testMLPRegressor(t, "identity", "square", "adam", 4)
}

func TestMLPRegressorLogisticLogLoss(t *testing.T) {
	testMLPRegressor(t, "logistic", "log", "adam", 2)
}

func TestMLPRegressorLogisticCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, "logistic", "cross-entropy", "adam", 2)
}

func TestMLPRegressorTanhCrossEntropyLoss(t *testing.T) {
	testMLPRegressor(t, "tanh", "cross-entropy", "adam", 2)
}

// func TestMLPRegressorReLUCrossEntropyLoss(t *testing.T) {
// 	testMLPRegressor(t, "relu", "cross-entropy", "adam", 2)
// }

func testMLPRegressor(t *testing.T, activationName string, lossName string, solver string, maxLayers int) {
	var nSamples, nFeatures, nOutputs = 2000, 3, 2
	activation := lm.Activations[activationName]
	var p = NewRandomProblem(nSamples, nFeatures, nOutputs, activationName, lossName)
	var HiddenLayerSizes []int

	for l := 0; l < maxLayers; l++ {
		Alpha := 1e-14
		regr := NewMLPRegressor(HiddenLayerSizes, activationName, solver, Alpha)
		// regr.SetOptimizer(func() Optimizer {
		// 	optimizer := base.NewAdamOptimizer()
		// 	optimizer.StepSize = 0.1
		// 	return optimizer
		// })

		//regr.SetOptimizer(OptimCreator, true)
		regr.Epochs = 400
		regr.GradientClipping = 5.
		testSetup := fmt.Sprintf("%T %T %s loss layers %v", regr, activation, lossName, HiddenLayerSizes)
		//Ypred := mat.NewDense(nSamples, nOutputs, nil)

		regr.Loss = lossName
		start := time.Now()
		regr.Fit(p.X, p.Y)
		elapsed := time.Since(start)
		unused(elapsed)

		if regr.J > 0.01 && regr.J > 0.5*regr.JFirst {
			t.Errorf("%s JFirst=%g J=%g", testSetup, regr.JFirst, regr.J)
		} else {
			//			fmt.Printf("%s ok J=%g elapsed=%s\n", testSetup, regr.J, elapsed)
		}
		HiddenLayerSizes = append(HiddenLayerSizes, 1+rand.Intn(9))
	}
}

func TestMLPClassifierMicrochip(t *testing.T) {
	X, Ytrue := datasets.LoadMicroChipTest()
	nSamples, nFeatures := X.Dims()

	//Xp, _ := preprocessing.NewPolynomialFeatures(6).Fit(X, Ytrue).Transform(X, Ytrue)
	// add poly features manually to have same order
	Xp := mat.NewDense(nSamples, 27, nil)
	c := 0
	for i := 1; i <= 6; i++ {
		for j := 0; j <= i; j++ {
			for s := 0; s < nSamples; s++ {
				Xp.Set(s, c, math.Pow(X.At(s, 0), float64(i-j))*math.Pow(X.At(s, 1), float64(j)))
			}
			c++
		}
	}

	_, nFeatures = Xp.Dims()
	_, nOutputs := Ytrue.Dims()
	regr := NewMLPClassifier([]int{}, "logistic", "adam", 1.)
	//regr.Loss = "cross-entropy"

	// we allocate Coef here because we use it for loss and grad tests before Fit
	regr.allocLayers(nFeatures, nOutputs, func() float64 { return 0. })

	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	var J float64
	loss := func() float64 {
		regr.predictZH(Xp, nil, Ypred, false)
		return regr.backprop(Xp, Ytrue)
	}
	chkLoss := func(context string, expectedLoss float64) {
		if math.Abs(J-expectedLoss) > 1e-3 {
			t.Errorf("%s J=%g expected:%g", context, J, expectedLoss)
		}
	}
	chkGrad := func(context string, expectedGradient []float64) {
		actualGradient := regr.Layers[0].Grad.RawRowView(0)[0:len(expectedGradient)]

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

	regr.Layers[0].Theta.Copy(base.MatApply0{Rows: 1 + nFeatures, Columns: nOutputs, Func: func() float64 { return 1 }})
	regr.Alpha = 10.

	J = loss()
	chkLoss("At test theta", 3.164)
	chkGrad("at test theta", []float64{0.3460, 0.1614, 0.1948, 0.2269, 0.0922})

	best := make(map[string]string)
	bestLoss := math.Inf(1)
	bestTime := time.Second * 86400

	// // test Fit with various base.Optimizer
	var Optimizers = []string{
		// "sgd",
		// "adagrad",
		// //"rmsprop",
		// "adadelta",
		"adam",
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
		regr.Layers[0].Theta.Apply(func(feature, output int, _ float64) float64 { return 0. }, regr.Layers[0].Theta)

		testSetup := optimizer
		start := time.Now()
		regr.Alpha = 1.
		regr.Epochs = 100
		regr.Layers[0].Optimizer = newOptimizer(optimizer)

		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()
		fmt.Println(testSetup, "elapsed time", elapsed, "loss", J)

		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		// Ypred.Apply(func(sample, output int, v float64) float64 {
		// 	if v >= .5 {
		// 		v = 1
		// 	} else {
		// 		v = 0
		// 	}
		// 	return v
		// }, Ypred)
		regr.Predict(Xp, Ypred)
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, nil, "").At(0, 0)
		expectedAccuracy := 0.83
		if accuracy < expectedAccuracy {
			t.Errorf("%s accuracy=%g expected:%g", optimizer, accuracy, expectedAccuracy)
		}
	}
	// fmt.Println("LogisticRegression BEST SETUP:", best)

	// // fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred, nil, "uniform_average").At(0, 0))
	// fmt.Println("ok")
}
