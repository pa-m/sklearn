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
	"github.com/pa-m/sklearn/preprocessing"

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
	testMLPRegressor(t, "identity", "square", "adam", 2)
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
	//activation := base.Activations[activationName]
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
		testSetup := fmt.Sprintf("%T %s %s loss layers %v", regr, activationName, lossName, HiddenLayerSizes)
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
		regr.predictZH(Xp, Ypred)
		return regr.fitEpoch(Xp, Ytrue, 1)
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
		"sgd",
		"adagrad",
		"rmsprop",
		"adadelta",
		"adam",
		"lbfgs",
	}
	newOptimizer := func(name string) base.Optimizer {

		switch name {
		case "adadelta":
			s := base.NewAdadeltaOptimizer()
			//s.StepSize = 0.1
			return s
		case "adam":
			s := base.NewAdamOptimizer()
			//s.StepSize = 2
			return s
		case "lbfgs":
			return nil
		default:
			s := base.NewOptimizer(name)
			return s
		}
	}
	//panic(fmt.Errorf("nSamples=%d", nSamples))
	for _, optimizer := range Optimizers {
		regr.Layers[0].Theta.Apply(func(feature, output int, _ float64) float64 { return 0. }, regr.Layers[0].Theta)

		testSetup := optimizer
		start := time.Now()
		regr.Alpha = 1.
		regr.Epochs = 50
		regr.MiniBatchSize = 118 //1,2,59,118
		regr.Solver = optimizer
		switch optimizer {
		case "lbfgs":
			regr.Layers[0].Optimizer = nil
		default:
			regr.Layers[0].Optimizer = newOptimizer(optimizer)
		}

		regr.Fit(Xp, Ytrue)
		elapsed := time.Since(start)
		J = loss()
		//fmt.Println(testSetup, "elapsed time", elapsed, "loss", J)

		if J < bestLoss {
			bestLoss = J
			best["best for loss"] = testSetup + fmt.Sprintf("(%g)", J)
		}
		if elapsed < bestTime {
			bestTime = elapsed
			best["best for time"] = testSetup + fmt.Sprintf("(%s)", elapsed)
		}
		regr.Predict(Xp, Ypred)
		accuracy := metrics.AccuracyScore(Ytrue, Ypred, true, nil)
		// FIXME accuracy should be over 0.83
		expectedAccuracy := 0.82
		if accuracy < expectedAccuracy {
			t.Errorf("%s accuracy=%g expected:%g", optimizer, accuracy, expectedAccuracy)
		}
	}
	fmt.Println("MLPClassifier BEST SETUP:", best)

	// // fmt.Println("acc:", metrics.AccuracyScore(Ytrue, Ypred,true,nil))
	// fmt.Println("ok")
}

func TestMnist(t *testing.T) {
	X, Y := datasets.LoadMnist()

	X, Yohe := preprocessing.NewOneHotEncoder().Fit(X, Y).Transform(X, Y)
	//fmt.Println(base.MatDimsString(Yohe))
	Theta1, Theta2 := datasets.LoadMnistWeights()
	mlp := NewMLPClassifier([]int{25}, "logistic", "adam", 0.)
	mlp.Loss = "cross-entropy"
	mlp.MiniBatchSize = 5000
	mlp.Shuffle = false
	mlp.allocLayers(400, 10, func() float64 { return 0. })
	mlp.Layers[0].Theta.Copy(Theta1.T())
	mlp.Layers[1].Theta.Copy(Theta2.T())
	J := mlp.fitEpoch(X, Yohe, 0)

	//fmt.Println("at test thetas J:=", J)
	// check cost at loaded theta is 0.287629
	if !floats.EqualWithinAbs(0.287629, J, 1e-6) {
		t.Errorf("Expected cost: %g, got %g", 0.287629, J)
	}

	mlp.Alpha = 1.
	mlp.Layers[0].Theta.Copy(Theta1.T())
	mlp.Layers[1].Theta.Copy(Theta2.T())
	J = mlp.fitEpoch(X, Yohe, 0)
	if !floats.EqualWithinAbs(0.383770, J, 1e-6) {
		t.Errorf("Expected cost: %g, got %g", 0.383770, J)
	}

}

func BenchmarkMnist(b *testing.B) {

	X, Y := datasets.LoadMnist()

	X, Yohe := preprocessing.NewOneHotEncoder().Fit(X, Y).Transform(X, Y)
	//fmt.Println(base.MatDimsString(Yohe))
	Theta1, Theta2 := datasets.LoadMnistWeights()
	mlp := NewMLPClassifier([]int{25}, "logistic", "adam", 0.)
	mlp.Loss = "cross-entropy"
	mlp.MiniBatchSize = 5000
	mlp.Shuffle = false
	mlp.allocLayers(400, 10, func() float64 { return 0. })
	mlp.Layers[0].Theta.Copy(Theta1.T())
	mlp.Layers[1].Theta.Copy(Theta2.T())
	mlp.fitEpoch(X, Yohe, 0)

	for epoch := 1; epoch < b.N*1; epoch++ {
		mlp.fitEpoch(X, Yohe, epoch)
	}

}

//go test ./neural_network -run BenchmarkMnist -bench ^BenchmarkMnist -cpuprofile /tmp/cpu.prof -memprofile /tmp/mem.prof -benchmem
//BenchmarkMnist-8   	      20	  62810454 ns/op	18984637 B/op	     201 allocs/op
//BenchmarkMnist-8   	      20	  62243278 ns/op	18985050 B/op	     202 allocs/op
//BenchmarkMnist-8   	      30	  41560266 ns/op	 2522941 B/op	     159 allocs/op
//BenchmarkMnist-8   	      30	  37128304 ns/op	 2522653 B/op	     156 allocs/op
//BenchmarkMnist-8   	      30	  35717715 ns/op	 2523106 B/op	     154 allocs/op
//BenchmarkMnist-8   	      50	  26804534 ns/op	 1550716 B/op	     117 allocs/op
