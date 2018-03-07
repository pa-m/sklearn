package neuralNetwork

import (
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/base"

	lm "github.com/pa-m/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// Optimizer comes from base
type Optimizer = base.Optimizer

// Layer represents a layer in a neural network. its mainly an Activation and a Theta
type Layer struct {
	Activation                                         string
	Theta, Ytrue, Z, Ypred, Ydiff, Hgrad, Grad, Update *mat.Dense
	Optimizer                                          Optimizer
}

// NewLayer creates a randomly initialized layer
func NewLayer(inputs, outputs int, activation string, optimizer Optimizer, thetaSlice []float64) Layer {

	Theta := mat.NewDense(inputs, outputs, thetaSlice)
	Theta.Apply(func(feature, output int, _ float64) float64 { return 0.01 * rand.Float64() }, Theta)
	return Layer{Activation: activation, Theta: Theta, Optimizer: optimizer}
}

// Init allocate matrices for layer
func (L *Layer) Init(samples, inputs int) {
	_, outputs := L.Theta.Dims()
	L.Z = mat.NewDense(samples, outputs, nil)
	L.Hgrad = mat.NewDense(samples, outputs, nil)
	L.Ypred = mat.NewDense(samples, outputs, nil)
	L.Ytrue = mat.NewDense(samples, outputs, nil)
	L.Ydiff = mat.NewDense(samples, outputs, nil)
	L.Grad = mat.NewDense(1+inputs, outputs, nil)
	L.Update = mat.NewDense(1+inputs, outputs, nil)
}

// Regressors is the list of regressors in this package
var Regressors = []lm.Regressor{&MLPRegressor{}}

// MLPRegressor is a multilayer perceptron regressor
type MLPRegressor struct {
	Optimizer        base.OptimCreator
	LossName         string
	Activation       string
	HiddenLayerSizes []int

	Layers                           []Layer
	Alpha, L1Ratio, GradientClipping float64
	Epochs, MiniBatchSize            int

	Loss string
	// run values
	thetaSlice []float64
	// Loss value after Fit
	JFirst, J float64
}

// OptimCreator is an Optimizer creator function
type OptimCreator = base.OptimCreator

// NewMLPRegressor returns a *MLPRegressor with defaults
// activation is on of lm.Identity{} lm.Logistic{} lm.Tanh{} lm.ReLU{} defaults to "relu"
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// lossName is one of square,log,cross-entropy (one of the keys of lm.LossFunctions)
func NewMLPRegressor(hiddenLayerSizes []int, activation string, solver string, Alpha float64) MLPRegressor {
	if activation == "" {
		activation = "relu"
	}
	if solver == "" {
		solver = "adam"
	}
	regr := MLPRegressor{
		Optimizer:        base.Solvers[solver],
		HiddenLayerSizes: hiddenLayerSizes,
		Loss:             "square",
		Activation:       activation,
		Alpha:            Alpha,
	}
	return regr
}

// NewMLPClassifier returns a *MLPRegressor with defaults
// activation is on of lm.Identity{} lm.Logistic{} lm.Tanh{} lm.ReLU{} defaults to "relu"
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// lossName is one of square,log,cross-entropy (one of the keys of lm.LossFunctions) defaults to "log"
func NewMLPClassifier(hiddenLayerSizes []int, activation string, solver string, Alpha float64) MLPRegressor {
	regr := NewMLPRegressor(hiddenLayerSizes, activation, solver, Alpha)
	regr.Loss = "log"
	return regr
}

// SetOptimizer changes Optimizer
func (regr *MLPRegressor) SetOptimizer(creator OptimCreator) {
	regr.Optimizer = creator
}

// Fit fits an MLPRegressor
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) lm.Regressor {
	_, nFeatures := X.Dims()
	_, nOutputs := Y.Dims()
	// create layers
	regr.Layers = make([]Layer, 0)
	prevOutputs := nFeatures
	var thetaLen, thetaOffset, thetaLen1 int
	for _, outputs := range regr.HiddenLayerSizes {
		thetaLen += (1 + prevOutputs) * outputs
		prevOutputs = outputs
	}
	thetaLen += (1 + prevOutputs) * nOutputs
	regr.thetaSlice = make([]float64, thetaLen, thetaLen)
	prevOutputs = nFeatures
	for _, outputs := range regr.HiddenLayerSizes {
		thetaLen1 = (1 + prevOutputs) * outputs
		regr.Layers = append(regr.Layers, NewLayer(1+prevOutputs, outputs, regr.Activation, regr.Optimizer(), regr.thetaSlice[thetaOffset:thetaOffset+thetaLen1]))
		thetaOffset += thetaLen1
		prevOutputs = outputs
	}
	var lastActivation string
	if regr.LossName == "cross-entropy" || regr.LossName == "log" {
		lastActivation = "logistic"
	} else {
		lastActivation = regr.Activation
	}
	// add output layer
	thetaLen1 = (1 + prevOutputs) * nOutputs
	regr.Layers = append(regr.Layers, NewLayer(1+prevOutputs, nOutputs, lastActivation, regr.Optimizer(), regr.thetaSlice[thetaOffset:thetaOffset+thetaLen1]))

	// J is the loss value
	regr.J = math.Inf(1)
	if regr.Epochs <= 0 {
		regr.Epochs = 100 // 1e6 / nSamples
	}
	for epoch := 0; epoch < regr.Epochs; epoch++ {
		regr.fitEpoch(X, Y, epoch)
	}
	return regr
}

func (regr *MLPRegressor) fitEpoch(Xfull, Yfull *mat.Dense, epoch int) {
	nSamples, nFeatures := Xfull.Dims()
	_, nOutputs := Yfull.Dims()
	base.DenseShuffle(Xfull, Yfull)
	miniBatchStart, miniBatchEnd, miniBatchLen := 0, nSamples, nSamples
	Jsum := 0.
	for miniBatchStart < nSamples {
		X := Xfull.Slice(miniBatchStart, miniBatchEnd, 0, nFeatures).(*mat.Dense)
		Y := Yfull.Slice(miniBatchStart, miniBatchEnd, 0, nOutputs).(*mat.Dense)
		Jmini := regr.fitMiniBatch(X, Y, epoch)
		Jsum += Jmini * float64(miniBatchLen)
		miniBatchStart, miniBatchEnd = miniBatchStart+miniBatchLen, miniBatchEnd+miniBatchLen
	}
	regr.J = Jsum / float64(nSamples)
	if epoch == 1 {
		regr.JFirst = Jsum / float64(nSamples)
	}
}

func (regr *MLPRegressor) fitMiniBatch(X, Y *mat.Dense, epoch int) float64 {
	regr.predictZH(X, nil, nil, true)
	return regr.backprop(X, Y)
}
func (regr *MLPRegressor) backprop(X, Y mat.Matrix) (J float64) {
	//nSamples, _ := X.Dims()
	outputLayer := len(regr.Layers) - 1
	//lossFunc := lm.LossFunctions[regr.Loss]

	for l := outputLayer; l >= 0; l-- {
		L := &regr.Layers[l]

		var Xl mat.Matrix
		if l == 0 {
			Xl = X
		} else {
			Xl = regr.Layers[l-1].Ypred
		}

		// compute Ydiff
		if l == outputLayer {
			L.Ytrue.Copy(Y)
			L.Ydiff.Sub(L.Ypred, Y)
		} else {
			// compute ydiff and ytrue for non-terminal layer
			//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
			nextLayer := &regr.Layers[l+1]
			//base.MatDimsCheck(".", L.Ydiff, nextLayer.Ydiff, base.MatFirstColumnRemoved{Matrix: nextLayer.Theta.T()})
			L.Ydiff.Mul(nextLayer.Ydiff, MatFirstColumnRemoved{Matrix: nextLayer.Theta.T()})
			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ydiff.MulElem(L.Ydiff, L.Hgrad)
			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ytrue.Sub(L.Ypred, L.Ydiff)
			//L.Ytrue.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ytrue)
		}

		// compute loss J and Grad
		//Jl := lossFunc(L.Ytrue, onesAddedMat{Xl}, L.Theta, L.Ypred, L.Ydiff, L.Grad, regr.Alpha, regr.L1Ratio, nSamples, L.Activation)
		lossfuncs := NewLoss(regr.Loss)
		Jl := lossfuncs.Loss(L.Ytrue, L.Ypred, L.Ydiff)
		// Ydiff is dJ/dH
		NewActivation(L.Activation).Grad(L.Z, L.Ypred, L.Hgrad)
		// =>L.Hgrad is derivative of activation vs Z

		L.Ydiff.MulElem(L.Ydiff, L.Hgrad)
		L.Grad.Mul(onesAddedMat{Matrix: Xl}.T(), L.Ydiff)

		if l == outputLayer {
			J = Jl
		}

		if regr.GradientClipping > 0 {
			GNorm := mat.Norm(L.Grad, 2)
			if GNorm > regr.GradientClipping {
				L.Grad.Scale(regr.GradientClipping/GNorm, L.Grad)
			}
		}
		//compute theeta Update from Grad
		L.Optimizer.GetUpdate(L.Update, L.Grad)
		// if l == outputLayer && epoch%10 == 0 {
		// 	fmt.Printf("epoch %d layer %d  J %g yt:%g yp:%g grad:%g upd:%g\n", epoch, l, J, L.Ytrue.At(0, 0), L.Ypred.At(0, 0), L.Grad.At(0, 0), L.Update.At(0, 0))
		// }
		L.Theta.Add(L.Theta, L.Update)
	}
	return J
}
func unused(...interface{}) {}

// Predict return the forward result
func (regr *MLPRegressor) Predict(X, Y *mat.Dense) lm.Regressor {
	regr.predictZH(X, nil, Y, false)
	return regr
}
func (regr *MLPRegressor) predictZH(X mat.Matrix, Z, Y *mat.Dense, fitting bool) lm.Regressor {
	outputLayer := len(regr.Layers) - 1
	for l := 0; l < len(regr.Layers); l++ {
		L := &regr.Layers[l]
		var Xl mat.Matrix
		if l == 0 {
			Xl = X
		} else {
			Xl = regr.Layers[l-1].Ypred
		}
		if L.Ypred == nil {
			samples, inputs := Xl.Dims()
			L.Init(samples, inputs)
		}
		if L.Ypred == nil {
			panic("L.Ypred == nil")
		}
		if regr.Layers[l].Ypred == nil {
			panic("L.Ypred == nil")
		}

		// compute activation.F([1 X] dot theta)
		L.Z.Mul(onesAddedMat{Matrix: Xl}, L.Theta)
		if l == outputLayer && Z != nil {
			Z.Copy(L.Z)
		}
		NewActivation(L.Activation).Func(L.Z, L.Ypred)
		L.Ypred.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ypred)
	}

	if Y != nil {
		Y.Copy(regr.Layers[len(regr.Layers)-1].Ypred)
	}
	return regr
}

// Score returns accuracy. see metrics package for other scores
func (regr *MLPRegressor) Score(X, Y *mat.Dense) float64 {
	score := 0.
	return score
}

func panicIfNaN(v float64) float64 {
	if math.IsNaN(v) {
		panic("NaN")
	}
	return v
}
