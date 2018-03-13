package neuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"pa-m/sklearn/metrics"

	"gonum.org/v1/gonum/optimize"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/blas"

	lm "github.com/pa-m/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// Optimizer comes from base
type Optimizer = base.Optimizer

// Layer represents a layer in a neural network. its mainly an Activation and a Theta
type Layer struct {
	Activation                                string
	X1, Ytrue, Z, Ypred, NextX1, Ydiff, Hgrad *mat.Dense
	Theta, Grad, Update                       *mat.Dense
	Optimizer                                 Optimizer
}

// NewLayer creates a randomly initialized layer
func NewLayer(inputs, outputs int, activation string, optimizer Optimizer, thetaSlice, gradSlice, updateSlice []float64, rnd func() float64) *Layer {

	Theta := mat.NewDense(inputs, outputs, thetaSlice)
	Theta.Apply(func(feature, output int, _ float64) float64 { return rnd() }, Theta)
	return &Layer{Activation: activation,
		Theta:     Theta,
		Grad:      mat.NewDense(inputs, outputs, gradSlice),
		Update:    mat.NewDense(inputs, outputs, updateSlice),
		Optimizer: optimizer}
}

func (L *Layer) allocOutputs(nSamples, nOutputs int) {
	mk := func(m **mat.Dense, nSamples, nOutputs int) {
		size := nSamples * nOutputs
		if *m == nil || cap((*m).RawMatrix().Data) < size {
			*m = mat.NewDense(nSamples, nOutputs, nil)
		} else {
			*m = mat.NewDense(nSamples, nOutputs, (*m).RawMatrix().Data[0:size])
		}
	}
	// make slices for Ytrue, Z, NextX, Ydiff, Hgrad
	mk(&L.Ytrue, nSamples, nOutputs)
	mk(&L.Z, nSamples, nOutputs)
	mk(&L.NextX1, nSamples, 1+nOutputs)
	mk(&L.Ydiff, nSamples, nOutputs)
	mk(&L.Hgrad, nSamples, nOutputs)

	for sample := 0; sample < nSamples; sample++ {
		L.NextX1.Set(sample, 0, 1.)
	}
	L.Ypred = base.MatDenseFirstColumnRemoved(L.NextX1)

}

// Regressors is the list of regressors in this package
var Regressors = []lm.Regressor{&MLPRegressor{}}

// MLPRegressor is a multilayer perceptron regressor
type MLPRegressor struct {
	Shuffle, UseBlas bool
	Optimizer        base.OptimCreator
	LossName         string
	Activation       string
	Solver           string
	HiddenLayerSizes []int

	Layers                           []*Layer
	Alpha, L1Ratio, GradientClipping float64
	Epochs, MiniBatchSize            int

	Loss string
	// run values
	thetaSlice, gradSlice, updateSlice []float64
	// Loss value after Fit
	JFirst, J float64
}

// OptimCreator is an Optimizer creator function
type OptimCreator = base.OptimCreator

// NewMLPRegressor returns a *MLPRegressor with defaults
// activation is one of identity,logistic,tanh,relu
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// Loss is one of square,log,cross-entropy defaults: square for identity, log for logistic,tanh,relu
func NewMLPRegressor(hiddenLayerSizes []int, activation string, solver string, Alpha float64) MLPRegressor {
	if activation == "" {
		activation = "relu"
	}
	if solver == "" {
		solver = "adam"
	}
	regr := MLPRegressor{
		Shuffle:          true,
		UseBlas:          true,
		Solver:           solver,
		Optimizer:        nil,
		HiddenLayerSizes: hiddenLayerSizes,
		Loss:             "square",
		Activation:       activation,
		Alpha:            Alpha,
	}
	if activation != "identity" {
		regr.Loss = "log"
	}
	switch {
	case isGOMethodOnly(solver):
	default:
		regr.SetOptimizer(base.Solvers[solver])
	}
	return regr
}

// SetOptimizer changes Optimizer
func (regr *MLPRegressor) SetOptimizer(creator OptimCreator) {
	regr.Optimizer = creator
}

func (regr *MLPRegressor) inputs(prevOutputs int) (inputs int) {
	inputs = 1 + prevOutputs
	return
}

func (regr *MLPRegressor) allocLayers(nFeatures, nOutputs int, rnd func() float64) {
	var thetaLen, thetaOffset, thetaLen1 int
	regr.Layers = make([]*Layer, 0)
	prevOutputs := nFeatures

	for _, outputs := range regr.HiddenLayerSizes {
		thetaLen += regr.inputs(prevOutputs) * outputs
		prevOutputs = outputs
	}
	thetaLen += regr.inputs(prevOutputs) * nOutputs
	regr.thetaSlice = make([]float64, thetaLen, thetaLen)
	regr.gradSlice = make([]float64, thetaLen, thetaLen)
	regr.updateSlice = make([]float64, thetaLen, thetaLen)
	prevOutputs = nFeatures
	for _, outputs := range regr.HiddenLayerSizes {
		thetaLen1 = regr.inputs(prevOutputs) * outputs
		regr.Layers = append(regr.Layers, NewLayer(regr.inputs(prevOutputs), outputs, regr.Activation, regr.Optimizer(),
			regr.thetaSlice[thetaOffset:thetaOffset+thetaLen1],
			regr.gradSlice[thetaOffset:thetaOffset+thetaLen1],
			regr.updateSlice[thetaOffset:thetaOffset+thetaLen1],
			rnd))
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
	thetaLen1 = regr.inputs(prevOutputs) * nOutputs
	regr.Layers = append(regr.Layers, NewLayer(1+prevOutputs, nOutputs, lastActivation, regr.Optimizer(),
		regr.thetaSlice[thetaOffset:thetaOffset+thetaLen1],
		regr.gradSlice[thetaOffset:thetaOffset+thetaLen1],
		regr.updateSlice[thetaOffset:thetaOffset+thetaLen1],
		rnd))

}

// Fit fits an MLPRegressor
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) lm.Regressor {
	nSamples, nFeatures := X.Dims()
	_, nOutputs := Y.Dims()
	// create layers
	regr.allocLayers(nFeatures, nOutputs, rand.NormFloat64)
	// J is the loss value
	regr.J = math.Inf(1)
	if regr.Epochs <= 0 {
		regr.Epochs = 1e6 / nSamples
	}
	switch {
	case isGOMethodOnly(regr.Solver):
		regr.fitGOM(X, Y)

	default:
		for epoch := 0; epoch < regr.Epochs; epoch++ {
			regr.fitEpoch(X, Y, epoch)
		}
	}
	return regr
}

// fitGOM fits with a gonum/optimize Method

func (regr *MLPRegressor) fitGOM(X, Y *mat.Dense) float64 {
	epoch := 0

	p := optimize.Problem{
		Func: func(thetaSlice []float64) float64 {
			copy(regr.thetaSlice, thetaSlice)
			J := regr.fitEpoch(X, Y, epoch)
			epoch++
			return J
		},
		Grad: func(gradSlice []float64, thetaSlice []float64) {
			copy(gradSlice, regr.gradSlice)
		},
	}
	method := base.GOMethodCreators[regr.Solver]()
	settings := optimize.DefaultSettings()
	settings.FuncEvaluations = regr.Epochs

	ret, err := optimize.Local(p, regr.thetaSlice, settings, method)
	if err != nil {
		fmt.Println(err)
	}
	copy(regr.thetaSlice, ret.X)
	regr.J = ret.F
	return ret.F
}

// fitEpoch fits one epoch
func (regr *MLPRegressor) fitEpoch(Xfull, Yfull *mat.Dense, epoch int) float64 {
	nSamples, nFeatures := Xfull.Dims()
	_, nOutputs := Yfull.Dims()
	if regr.Shuffle {
		base.MatShuffle(Xfull, Yfull)
	}
	var miniBatchSize int
	switch {
	case regr.MiniBatchSize > 0 && regr.MiniBatchSize <= nSamples:
		miniBatchSize = regr.MiniBatchSize
	case regr.MiniBatchSize > nSamples:
		miniBatchSize = nSamples
	default:
		miniBatchSize = nSamples
		if miniBatchSize > 200 {
			miniBatchSize = 200
		}
	}
	miniBatchStart, miniBatchEnd := 0, miniBatchSize
	Jsum := 0.
	for miniBatchStart < nSamples {
		miniBatchLen := miniBatchEnd - miniBatchStart
		X := base.MatDenseSlice(Xfull, miniBatchStart, miniBatchEnd, 0, nFeatures)
		Y := base.MatDenseSlice(Yfull, miniBatchStart, miniBatchEnd, 0, nOutputs)

		Jmini := regr.fitMiniBatch(X, Y, epoch, miniBatchLen, nSamples)
		Jsum += Jmini
		miniBatchStart, miniBatchEnd = miniBatchStart+miniBatchLen, miniBatchEnd+miniBatchLen
		if miniBatchEnd > nSamples {
			miniBatchEnd = nSamples
		}
	}
	regr.J = Jsum
	if epoch == 1 {
		regr.JFirst = Jsum
	}
	return Jsum
}

// fitMiniBatch fit one minibatch
func (regr *MLPRegressor) fitMiniBatch(Xmini, Ymini *mat.Dense, epoch, miniBatchLen, nSamples int) float64 {
	regr.predictZH(Xmini, nil)
	Jmini := regr.backprop(Xmini, Ymini, epoch, miniBatchLen, nSamples)
	return Jmini
}

// backprop corrects weights
func (regr *MLPRegressor) backprop(X, Y mat.Matrix, epoch, miniBatchLen, nSamples int) (J float64) {
	//nSamples, _ := X.Dims()
	miniBatchPart := float64(miniBatchLen) / float64(nSamples)
	_, nOutputs := Y.Dims()
	outputLayer := len(regr.Layers) - 1

	J = 0
	for l := outputLayer; l >= 0; l-- {
		L := regr.Layers[l]

		// compute Ydiff
		if l == outputLayer {
			L.Ytrue.Copy(Y)
			L.Ydiff.Sub(L.Ypred, Y)
		} else {
			// compute ydiff and ytrue for non-terminal layer
			//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
			nextLayer := regr.Layers[l+1]

			if regr.UseBlas {
				NextThetaG := nextLayer.Theta.RawMatrix()
				NextThetaG1 := base.MatDenseSlice(nextLayer.Theta, 1, NextThetaG.Rows, 0, NextThetaG.Cols)
				//  C = alpha * A * B + beta * C
				base.MatParallelGemm(blas.NoTrans, blas.Trans, 1., nextLayer.Ydiff.RawMatrix(), NextThetaG1.RawMatrix(), 0., L.Ydiff.RawMatrix())

			} else {
				L.Ydiff.Mul(nextLayer.Ydiff, base.MatFirstColumnRemoved{Matrix: nextLayer.Theta.T()})

			}

			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ydiff.MulElem(rm(L.Ydiff), rm(L.Hgrad))
			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ytrue.Sub(rm(L.Ypred), rm(L.Ydiff))
			//L.Ytrue.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ytrue)
		}

		// compute loss J and Grad, put loss gradient in Ydiff
		if l == outputLayer {
			lastLoss := regr.Loss
			if lastLoss == "log" && nOutputs == 1 {
				lastLoss = "cross-entropy"
			}
			J = NewLoss(lastLoss).Loss(L.Ytrue, L.Ypred, L.Ydiff) * miniBatchPart
		} else {
			NewLoss("square").Loss(L.Ytrue, L.Ypred, L.Ydiff)
		}

		// Ydiff is dJ/dH
		NewActivation(L.Activation).Grad(L.Z, L.Ypred, L.Hgrad)
		// =>L.Hgrad is derivative of activation vs Z

		// put dJ/dh*dh/dz in Ydiff
		L.Ydiff.MulElem(L.Ydiff, L.Hgrad)

		// put [1 X].T * (dJ/dh.*dh/dz) in L.Grad
		if regr.UseBlas {
			base.MatParallelGemm(blas.Trans, blas.NoTrans, 1., L.X1.RawMatrix(), L.Ydiff.RawMatrix(), 0., L.Grad.RawMatrix())

		} else {
			L.Grad.Mul(L.X1.T(), L.Ydiff)
		}

		Alpha := regr.Alpha
		// Add regularization to cost and grad
		if Alpha > 0. {
			L1Ratio := regr.L1Ratio
			R := 0.
			features, outputs := L.Theta.Dims()
			//ThetaReg := base.MatFirstRowZeroed{Matrix: L.Theta}
			ThetaReg := base.MatDenseSlice(L.Theta, 1, features, 0, outputs)
			GradReg := base.MatDenseSlice(L.Grad, 1, features, 0, outputs)
			if L1Ratio > 0. {
				// add L1 regularization
				//R += Alpha * L1Ratio / float64(nSamples) * mat.Sum(matApply{Matrix: ThetaReg, Func: math.Abs})
				R += Alpha * L1Ratio / float64(nSamples) * matx{Dense: ThetaReg}.SumAbs()
				//GradReg.Add(GradReg, matScale{Matrix: matApply{Matrix: ThetaReg, Func: sgn}, Scale: Alpha / float64(nSamples)})
				matx{Dense: GradReg}.AddScaledApplied(Alpha/float64(nSamples), ThetaReg, sgn)
			}
			if L1Ratio < 1. {
				// add L2 regularization
				R += Alpha * (1. - L1Ratio) / 2. / float64(nSamples) * matx{Dense: ThetaReg}.SumSquares()
				//GradReg.Add(GradReg, matScale{Matrix: ThetaReg, Scale: Alpha / float64(nSamples)})
				matx{Dense: GradReg}.AddScaled(Alpha/float64(nSamples), ThetaReg)
			}
			J += R
		}

		if regr.GradientClipping > 0. {
			GNorm := mat.Norm(L.Grad, 2.)
			if GNorm > regr.GradientClipping {
				L.Grad.Scale(regr.GradientClipping/GNorm, L.Grad)
			}
		}
		//compute theta Update from Grad
		switch regr.Solver {
		case "lbfgs":
		default:
			L.Optimizer.GetUpdate(L.Update, L.Grad)
			L.Theta.Add(L.Theta, L.Update)
		}
	}
	return J
}

func unused(...interface{}) {}

// Predict return the forward result
func (regr *MLPRegressor) Predict(X, Y *mat.Dense) lm.Regressor {
	regr.predictZH(X, Y)
	return regr
}

// put X dot Theta in Z and activation(X dot Theta) in Y
// Z and Y can be nil
func (regr *MLPRegressor) predictZH(X, Y *mat.Dense) lm.Regressor {
	nSamples, nFeatures0 := X.Dims()
	for l := 0; l < len(regr.Layers); l++ {
		L := regr.Layers[l]
		_, nOutputs := L.Theta.Dims()
		L.allocOutputs(nSamples, nOutputs)
		if l == 0 {
			if L.X1 == nil || len(L.X1.RawMatrix().Data) != (nSamples*(1+nFeatures0)) {
				L.X1 = mat.NewDense(nSamples, 1+nFeatures0, nil)
			}
			//L.X1.Copy(onesAddedMat{Matrix: X})
			matx{Dense: L.X1}.CopyPrependOnes(X)
		} else {
			L.X1 = regr.Layers[l-1].NextX1
		}

		if L.Ypred == nil {
			panic("L.Ypred == nil")
		}
		if regr.Layers[l].Ypred == nil {
			panic("L.Ypred == nil")
		}

		// compute activation.F([1 X] dot theta)
		if regr.UseBlas {
			base.MatParallelGemm(blas.NoTrans, blas.NoTrans, 1., L.X1.RawMatrix(), L.Theta.RawMatrix(), 0., L.Z.RawMatrix())
		} else {
			L.Z.Mul(L.X1, L.Theta)
		}

		NewActivation(L.Activation).Func(L.Z, L.Ypred)
	}

	if Y != nil {
		Y.Copy(regr.Layers[len(regr.Layers)-1].Ypred)
	}
	return regr
}

// Score returns accuracy. see metrics package for other scores
func (regr *MLPRegressor) Score(X, Y *mat.Dense) float64 {
	nSamples, _ := Y.Dims()
	_, nOutputs := Y.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	regr.Predict(X, Y)
	if regr.Loss == "square" {
		return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
	}
	return metrics.AccuracyScore(Y, Ypred, nil, "").At(0, 0)
}

// MLPClassifier ...
type MLPClassifier struct{ MLPRegressor }

// NewMLPClassifier returns a *MLPClassifier with defaults
// activation is one of logistic,tanh,relu
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// lossName is one of square,log,cross-entropy (one of the keys of lm.LossFunctions) defaults to "log"
func NewMLPClassifier(hiddenLayerSizes []int, activation string, solver string, Alpha float64) MLPClassifier {
	regr := MLPClassifier{
		MLPRegressor: NewMLPRegressor(hiddenLayerSizes, activation, solver, Alpha),
	}
	regr.Loss = "log"
	return regr
}

// Predict return the forward result for MLPClassifier
func (regr *MLPClassifier) Predict(X, Y *mat.Dense) lm.Regressor {
	regr.predictZH(X, Y)
	Y.Copy(regr.Layers[len(regr.Layers)-1].Ypred)
	Y.Apply(func(i, o int, y float64) float64 {
		if y >= .5 {
			y = 1.
		} else {
			y = 0.
		}
		return y
	}, Y)
	return regr
}

func panicIfNaN(v float64) float64 {
	if math.IsNaN(v) {
		panic("NaN")
	}
	return v
}

func sgn(x float64) float64 {
	if x > 0. {
		return 1.
	}
	if x < 0. {
		return -1.
	}
	return 0.
}

// rm checks arg is a *Dense and returns it
func rm(m *mat.Dense) *mat.Dense { return m }

func isGOMethodOnly(solver string) bool {
	_, isBaseOptimCreator := base.Solvers[solver]
	_, isGOMethodCreator := base.GOMethodCreators[solver]
	return isGOMethodCreator && !isBaseOptimCreator
}
