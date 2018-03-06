package neuralNetwork

import (
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/base"

	lm "github.com/pa-m/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// Activation comes from linear_model
type Activation = lm.Activation

// Optimizer comes from base
type Optimizer = base.Optimizer

//Loss commes from linear_model
type Loss = lm.Loss

// Layer represents a layer in a neural network. its mainly an Activation and a Theta
type Layer struct {
	Activation
	Theta, Ytrue, Ypred, Ydiff, Grad, Update *mat.Dense
	Optimizer                                Optimizer
}

// NewLayer creates a randomly initialized layer
func NewLayer(inputs, outputs int, activation lm.Activation, optimizer Optimizer) Layer {
	Theta := mat.NewDense(inputs, outputs, nil)
	Theta.Apply(func(feature, output int, _ float64) float64 { return 0.01 * rand.Float64() }, Theta)
	return Layer{Activation: activation, Theta: Theta, Optimizer: optimizer}
}

// Init allocate matrices for layer
func (L *Layer) Init(samples, inputs int) {
	_, outputs := L.Theta.Dims()
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
	Optimizer func() base.Optimizer
	lm.Loss
	Layers                []Layer
	Alpha, L1Ratio        float64
	Epochs, MiniBatchSize int
	// Loss value after Fit
	JFirst, J float64
}

// OptimCreator is an Optimizer creator function
type OptimCreator func() Optimizer

// NewMLPRegressor returns a *MLPRegressor with defaults
func NewMLPRegressor(hiddenLayerSizes []int, activation lm.Activation, Optimiser OptimCreator) MLPRegressor {
	if Optimiser == nil {
		Optimiser = func() Optimizer { return base.NewAdamOptimizer() }
	}
	regr := MLPRegressor{
		Optimizer: Optimiser,
		Loss:      lm.SquareLoss,
	}
	prevOutputs := 1
	for _, outputs := range hiddenLayerSizes {
		regr.Layers = append(regr.Layers, NewLayer(1+prevOutputs, outputs, activation, regr.Optimizer()))
		prevOutputs = outputs
	}
	regr.Layers = append(regr.Layers, NewLayer(1+prevOutputs, 1, activation, regr.Optimizer()))
	return regr
}

// SetOptimizer changes Optimizer
func (regr *MLPRegressor) SetOptimizer(creator OptimCreator, changeLayers bool) {
	regr.Optimizer = creator
	if changeLayers {
		for l := 0; l < len(regr.Layers); l++ {
			regr.Layers[l].Optimizer = creator()
		}
	}
}

// Fit fits an MLPRegressor
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) lm.Regressor {
	nSamples, nFeatures := X.Dims()
	outputLayer := len(regr.Layers) - 1
	r, c := regr.Layers[0].Theta.Dims()

	// adjust size of first layer to X features
	if r != nFeatures+1 {
		regr.Layers[0] = NewLayer(1+nFeatures, c, regr.Layers[0].Activation, regr.Optimizer())
	}
	// adjust size of output layer to Y outputs
	r, c = Y.Dims()
	r2, c2 := regr.Layers[outputLayer].Theta.Dims()
	if c != c2 {
		regr.Layers[outputLayer] = NewLayer(r2, c, regr.Layers[0].Activation, regr.Optimizer())
	}
	// J is the loss value
	J := math.Inf(1)
	if regr.Epochs <= 0 {
		regr.Epochs = 100 // 1e6 / nSamples
	}
	for epoch := 0; epoch < regr.Epochs; epoch++ {
		base.DenseShuffle(X, Y)
		regr.Predict(X, nil)
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
				L.Ytrue.Clone(Y)
				L.Ydiff.Sub(L.Ypred, Y)
			} else {
				// compute ydiff and ytrue for non-terminal layer
				//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
				nextLayer := &regr.Layers[l+1]

				L.Ydiff.Mul(nextLayer.Ydiff, firstColumnRemovedMat{nextLayer.Theta.T()})
				L.Ydiff.MulElem(L.Ydiff, appliedMat{L.Ypred, L.Activation.Fprime})
				L.Ytrue.Sub(L.Ypred, L.Ydiff)
			}
			// compute loss J and Grad
			J = regr.Loss(L.Ytrue, onesAddedMat{Xl}, L.Theta, L.Ypred, L.Ydiff, L.Grad, regr.Alpha, regr.L1Ratio, nSamples, L.Activation)
			//compute theeta Update from Grad
			L.Optimizer.GetUpdate(L.Update, L.Grad)
			// if l == outputLayer && epoch%10 == 0 {
			// 	fmt.Printf("epoch %d layer %d  J %g yt:%g yp:%g grad:%g upd:%g\n", epoch, l, J, L.Ytrue.At(0, 0), L.Ypred.At(0, 0), L.Grad.At(0, 0), L.Update.At(0, 0))
			// }
			L.Theta.Add(L.Theta, L.Update)
			if l == outputLayer {
				regr.J = J
				if epoch == 1 {
					regr.JFirst = J
				}
			}
		}
	}
	return regr
}

func unused(...interface{}) {}

// Predict return the forward result
func (regr *MLPRegressor) Predict(X, Y *mat.Dense) lm.Regressor {
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
		L.Ypred.Mul(onesAddedMat{Xl}, L.Theta)
		L.Ypred.Clone(appliedMat{L.Ypred, L.Activation.F})
	}
	if Y != nil {
		Y.Clone(regr.Layers[len(regr.Layers)-1].Ypred)
	}
	return regr
}

// Score returns accuracy. see metrics package for other scores
func (regr *MLPRegressor) Score(X, Y *mat.Dense) float64 {
	score := 0.
	return score
}
