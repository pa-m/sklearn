package neuralNetwork

import (
	"math"
	"math/rand"
	"pa-m/sklearn/metrics"

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

	slices              struct{ Ytrue, Z, NextX1, Ydiff, Hgrad []float64 }
	Theta, Grad, Update *mat.Dense
	Optimizer           Optimizer
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
	s := &L.slices
	mk := func(s *[]float64, m **mat.Dense, nSamples, nOutputs int) {
		size := nSamples * nOutputs
		if *s == nil || cap(*s) < size {
			*s = make([]float64, size)
		} else {
			*s = (*s)[0:size]
		}
		*m = mat.NewDense(nSamples, nOutputs, *s)
	}
	// make slices for Ytrue, Z, NextX, Ydiff, Hgrad
	mk(&s.Ytrue, &L.Ytrue, nSamples, nOutputs)
	mk(&s.Z, &L.Z, nSamples, nOutputs)
	mk(&s.NextX1, &L.NextX1, nSamples, 1+nOutputs)
	mk(&s.Ydiff, &L.Ydiff, nSamples, nOutputs)
	mk(&s.Hgrad, &L.Hgrad, nSamples, nOutputs)
	//mk(&s.Ypred, &L.Ypred, nSamples, nOutputs)

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
		Optimizer:        base.Solvers[solver],
		HiddenLayerSizes: hiddenLayerSizes,
		Loss:             "square",
		Activation:       activation,
		Alpha:            Alpha,
	}
	if activation != "identity" {
		regr.Loss = "log"
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
	for epoch := 0; epoch < regr.Epochs; epoch++ {
		regr.fitEpoch(X, Y, epoch)
	}
	return regr
}

// firEpoch fits one epoch
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
		//fmt.Printf("miniBatchStart %d, miniBatchEnd %d\n", miniBatchStart, miniBatchEnd)
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
	regr.predictZH(Xmini, nil, nil, true)
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
			//base.MatDimsCheck("-", L.Ydiff, L.Ypred, Y)
			L.Ydiff.Sub(L.Ypred, Y)
		} else {
			// compute ydiff and ytrue for non-terminal layer
			//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
			nextLayer := regr.Layers[l+1]
			//base.MatDimsCheck(".", L.Ydiff, nextLayer.Ydiff, base.MatFirstColumnRemoved{Matrix: nextLayer.Theta.T()})

			if regr.UseBlas {
				NextThetaG := nextLayer.Theta.RawMatrix()
				NextThetaG1 := base.MatDenseSlice(nextLayer.Theta, 1, NextThetaG.Rows, 0, NextThetaG.Cols)
				//  C = alpha * A * B + beta * C
				base.MatParallelGemm(blas.NoTrans, blas.Trans, 1., nextLayer.Ydiff.RawMatrix(), NextThetaG1.RawMatrix(), 0., L.Ydiff.RawMatrix())

			} else {
				L.Ydiff.Mul(nextLayer.Ydiff, MatFirstColumnRemoved{Matrix: nextLayer.Theta.T()})

			}

			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ydiff.MulElem(L.Ydiff, L.Hgrad)
			//L.Ydiff.Apply(func(_, _ int, v float64) float64 { return panicIfNaN(v) }, L.Ydiff)
			L.Ytrue.Sub(L.Ypred, L.Ydiff)
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
			ThetaReg := base.MatFirstRowZeroed{Matrix: L.Theta}
			if L1Ratio > 0. {
				// add L1 regularization
				R += Alpha * L1Ratio / float64(nSamples) * mat.Sum(matApply{Matrix: ThetaReg, Func: math.Abs})
				L.Grad.Add(L.Grad, matScale{Matrix: matApply{Matrix: ThetaReg, Func: sgn}, Scale: Alpha / float64(nSamples)})
			}
			if L1Ratio < 1. {
				// add L2 regularization
				R += Alpha * (1. - L1Ratio) / 2. / float64(nSamples) * mat.Sum(matMulElem{A: ThetaReg, B: ThetaReg})
				L.Grad.Add(L.Grad, matScale{Matrix: ThetaReg, Scale: Alpha / float64(nSamples)})
			}
			//fmt.Println("J", J, "R", R, fmt.Sprintf(" %g*%g/%d*%g", Alpha, 1.-L1Ratio, 2*nSamples, mat.Sum(matMulElem{A: ThetaReg, B: ThetaReg})))
			J += R
		}

		if regr.GradientClipping > 0. {
			GNorm := mat.Norm(L.Grad, 2.)
			if GNorm > regr.GradientClipping {
				L.Grad.Scale(regr.GradientClipping/GNorm, L.Grad)
			}
		}
		//compute theta Update from Grad
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

// put X dot Theta in Z and activation(X dot Theta) in Y
// Z and Y can be nil
func (regr *MLPRegressor) predictZH(X, Z, Y *mat.Dense, fitting bool) lm.Regressor {
	nSamples, nFeatures0 := X.Dims()
	outputLayer := len(regr.Layers) - 1
	for l := 0; l < len(regr.Layers); l++ {
		L := regr.Layers[l]
		_, nOutputs := L.Theta.Dims()
		L.allocOutputs(nSamples, nOutputs)
		if l == 0 {
			L.X1 = mat.NewDense(nSamples, 1+nFeatures0, nil)
			L.X1.Copy(onesAddedMat{Matrix: X})
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
		//base.MatDimsCheck(".", L.Z, onesAddedMat{Matrix: Xl}, L.Theta)
		if regr.UseBlas {
			L.Z.Mul(L.X1, L.Theta)
		} else {
			base.MatParallelGemm(blas.NoTrans, blas.NoTrans, 1., L.X1.RawMatrix(), L.Theta.RawMatrix(), 0., L.Z.RawMatrix())
		}

		if l == outputLayer && Z != nil {
			Z.Copy(L.Z)
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
	regr.predictZH(X, nil, Y, false)
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
