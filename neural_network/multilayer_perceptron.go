package neuralNetwork

import (
	"fmt"
	"math"
	"math/rand"
	"pa-m/sklearn/base"

	lm "github.com/pa-m/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

// Layer represents a layer in a neural network. its mainly an Activation and a Theta
type Layer struct {
	lm.Activation
	Theta, Ytrue, Ypred, Ydiff, Grad, Update *mat.Dense
}

// NewLayer creates a randomly initialized layer
func NewLayer(inputs, outputs int, activation lm.Activation) Layer {
	Theta := mat.NewDense(inputs, outputs, nil)
	Theta.Apply(func(feature, output int, _ float64) float64 { return 0.01 * rand.Float64() }, Theta)
	return Layer{Activation: activation, Theta: Theta}
}

// Regressors is the list of regressors in this package
var Regressors = []lm.Regressor{&MLPRegressor{}}

// MLPRegressor is a multilayer perceptron regressor
type MLPRegressor struct {
	Optimizer base.Optimizer
	lm.Loss
	Layers                []Layer
	Alpha, L1Ratio        float64
	Epochs, MiniBatchSize int
}

// NewMLPRegressor returns a *MLPRegressor with defaults
func NewMLPRegressor(hiddenLayerSizes []int, activation lm.Activation, solver base.Optimizer) MLPRegressor {
	regr := MLPRegressor{Optimizer: solver, Loss: lm.SquareLoss}
	prevOutputs := 1
	for _, outputs := range hiddenLayerSizes {
		regr.Layers = append(regr.Layers, NewLayer(prevOutputs, outputs, activation))
		prevOutputs = outputs
	}
	return regr
}

// Fit fits an MLPRegressor
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) lm.Regressor {
	nSamples, nFeatures := X.Dims()
	r, c := regr.Layers[0].Theta.Dims()

	if r != nFeatures+1 {
		regr.Layers[0] = NewLayer(nFeatures+1, c, regr.Layers[0].Activation)
	}
	J := math.Inf(1)
	if regr.Epochs <= 0 {
		regr.Epochs = 1e6 / nSamples
	}
	for epoch := 0; epoch < regr.Epochs; epoch++ {
		for l := 0; l < len(regr.Layers); l++ {
			L := regr.Layers[l]
			var Xl mat.Matrix
			if l == 0 {
				Xl = X
			} else {
				Xl = regr.Layers[l-1].Ypred
			}
			// compute [1 X] dot theta
			L.Ypred.Apply(func(sample, output int, _ float64) float64 {
				var xl, ypred float64
				_, nFeatures := L.Theta.Dims()
				for input := 0; input < nFeatures; input++ {
					if input == 0 {
						xl = 1.
					} else {
						xl = Xl.At(sample, input-1)
					}
					ypred += xl * L.Theta.At(input, output)
				}
				return L.Activation.F(ypred)
			}, L.Ypred)
		}
		J = mat.Norm(regr.Layers[len(regr.Layers)-1].Ydiff, 2)
		J = J * J / float64(nSamples)

		fmt.Println(epoch, J)
		for l := len(regr.Layers) - 1; l > 0; l-- {
			L := regr.Layers[l]
			var Xl mat.Matrix
			if l == 0 {
				Xl = X
			} else {
				Xl = regr.Layers[l-1].Ypred
			}
			// compute Ydiff
			if l == len(regr.Layers)-1 {
				L.Ytrue.Clone(Y)
				L.Ydiff.Sub(L.Ypred, L.Ytrue)
				//compute Ydiff.T dot [1 prev_Ypred]
				L.Update.Apply(func(input, output int, _ float64) float64 {
					upd := 0.
					var xl float64

					for sample := 0; sample < nSamples; sample++ {
						if input == 0 {
							xl = 1
						} else {
							xl = Xl.At(sample, input-1)
						}
						upd += L.Ydiff.At(sample, output) * xl
					}
					return upd
				}, L.Update)
			} else {
				//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
				L.Ydiff.Apply(func(sample, output int, _ float64) float64 {
					_, nextOutputs := regr.Layers[l+1].Ydiff.Dims()
					ydiff := 0.
					for on := 0; on < nextOutputs; on++ {
						ydiff += regr.Layers[l+1].Ydiff.At(sample, output+1) * regr.Layers[l+1].Theta.At(output+1, on)
					}
					return ydiff
				}, L.Ydiff)
				L.Ytrue.Add(L.Ypred, L.Ydiff)
				L.Update.Apply(func(input, output int, _ float64) float64 {
					upd := 0.
					var xl float64
					for sample := 0; sample < nSamples; sample++ {
						if output == 0 {
							xl = 1.
						} else {
							xl = Xl.At(sample, output-1)
						}
						upd += L.Ydiff.At(sample, output) * xl
					}
					return upd
				}, L.Update)
			}
			L.Update.Scale(1./float64(nSamples), L.Update)
			L.Grad.Add(L.Grad, L.Update)
		}
	}
	return regr
}

func unused(...interface{}) {}

// Predict return the forward result
func (regr *MLPRegressor) Predict(X, Y *mat.Dense) lm.Regressor {
	for l := 0; l < len(regr.Layers); l++ {
		L := regr.Layers[l]
		var Xl mat.Matrix
		if l == 0 {
			Xl = X
		} else {
			Xl = regr.Layers[l-1].Ypred
		}
		if L.Ypred == nil {
			samples, inputs := Xl.Dims()
			_, outputs := L.Theta.Dims()
			L.Ypred = mat.NewDense(samples, outputs, nil)
			L.Ydiff = mat.NewDense(samples, outputs, nil)
			L.Grad = mat.NewDense(1+inputs, outputs, nil)
			L.Update = mat.NewDense(1+inputs, outputs, nil)
		}
		// compute [1 X] dot theta
		L.Ypred.Apply(func(sample, output int, _ float64) float64 {
			var xl, ypred float64
			_, nFeatures := L.Theta.Dims()
			for input := 0; input < nFeatures; input++ {
				if input == 0 {
					xl = 1.
				} else {
					xl = Xl.At(sample, input-1)
				}
				ypred += xl * L.Theta.At(input, output)
			}
			return L.Activation.F(ypred)
		}, L.Ypred)
	}
	Y.Clone(regr.Layers[len(regr.Layers)-1].Ypred)
	return regr
}

// Score returns accuracy. see metrics package for other scores
func (regr *MLPRegressor) Score(X, Y *mat.Dense) float64 {
	score := 0.
	return score
}
