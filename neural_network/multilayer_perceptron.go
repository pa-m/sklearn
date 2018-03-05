package neuralNetwork

import (
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
	Optimizer                                base.Optimizer
}

// NewLayer creates a randomly initialized layer
func NewLayer(inputs, outputs int, activation lm.Activation, optimizer base.Optimizer) Layer {
	Theta := mat.NewDense(inputs, outputs, nil)
	Theta.Apply(func(feature, output int, _ float64) float64 { return 0.01 * rand.Float64() }, Theta)
	return Layer{Activation: activation, Theta: Theta, Optimizer: optimizer}
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
	J float64
}

// NewMLPRegressor returns a *MLPRegressor with defaults
func NewMLPRegressor(hiddenLayerSizes []int, activation lm.Activation, Optimiser func() base.Optimizer) MLPRegressor {
	if Optimiser == nil {
		Optimiser = func() base.Optimizer { return base.NewAdamOptimizer() }
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

// Fit fits an MLPRegressor
func (regr *MLPRegressor) Fit(X, Y *mat.Dense) lm.Regressor {
	nSamples, nFeatures := X.Dims()
	outputLayer := len(regr.Layers) - 1
	r, c := regr.Layers[0].Theta.Dims()

	if r != nFeatures+1 {
		regr.Layers[0] = NewLayer(nFeatures+1, c, regr.Layers[0].Activation, regr.Optimizer())
	}
	r, c = Y.Dims()
	r2, c2 := regr.Layers[outputLayer].Theta.Dims()
	if c != c2 {
		regr.Layers[outputLayer] = NewLayer(r2, c, regr.Layers[0].Activation, regr.Optimizer())
	}
	J := math.Inf(1)
	if regr.Epochs <= 0 {
		regr.Epochs = 100 // 1e6 / nSamples
	}
	for epoch := 0; epoch < regr.Epochs; epoch++ {
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
				J = regr.Loss(L.Ytrue, onesAddedMat{X: Xl}, L.Theta, L.Ypred, L.Ydiff, L.Grad, regr.Alpha, regr.L1Ratio, nSamples, L.Activation)
				//J = mat.Norm(L.Ydiff, 2)
			} else {
				//delta2 = (delta3 * Theta2) .* [1 a2(t,:)] .* (1-[1 a2(t,:)])
				L.Ydiff.Apply(func(sample, output int, _ float64) float64 {
					_, nextOutputs := regr.Layers[l+1].Ydiff.Dims()
					ydiff := 0.
					yp := L.Ypred.At(sample, output)
					for nextOut := 0; nextOut < nextOutputs; nextOut++ {
						nextFeat := output + 1
						nextYd := regr.Layers[l+1].Ydiff.At(sample, nextOut)
						nextTh := regr.Layers[l+1].Theta.At(nextFeat, nextOut)
						ydiff += nextYd *
							nextTh *
							yp * (1. - yp)
					}
					return ydiff
				}, L.Ydiff)
				L.Ytrue.Sub(L.Ypred, L.Ydiff)

				//J = mat.Norm(L.Ydiff, 2)
				J = regr.Loss(L.Ytrue, onesAddedMat{X: Xl}, L.Theta, L.Ypred, L.Ydiff, L.Grad, regr.Alpha, regr.L1Ratio, nSamples, L.Activation)

			}
			//compute Ydiff.T dot [1 prev_Ypred]
			/*L.Update.Apply(func(input, output int, _ float64) float64 {
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
			L.Update.Scale(-1./float64(nSamples), L.Update)
			*/
			L.Optimizer.GetUpdate(L.Update, L.Grad)
			// if l == outputLayer && epoch%10 == 0 {
			// 	fmt.Printf("epoch %d layer %d  J %g yt:%g yp:%g grad:%g upd:%g\n", epoch, l, J, L.Ytrue.At(0, 0), L.Ypred.At(0, 0), L.Grad.At(0, 0), L.Update.At(0, 0))
			// }

			L.Theta.Add(L.Theta, L.Update)
			regr.J = J
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
			_, outputs := L.Theta.Dims()
			L.Ypred = mat.NewDense(samples, outputs, nil)
			L.Ytrue = mat.NewDense(samples, outputs, nil)
			L.Ydiff = mat.NewDense(samples, outputs, nil)
			L.Grad = mat.NewDense(1+inputs, outputs, nil)
			L.Update = mat.NewDense(1+inputs, outputs, nil)
		}
		if L.Ypred == nil {
			panic("L.Ypred == nil")
		}
		if regr.Layers[l].Ypred == nil {
			panic("L.Ypred == nil")
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

type onesAddedMat struct{ X mat.Matrix }

func (m onesAddedMat) Dims() (int, int) { r, c := m.X.Dims(); return r, 1 + c }
func (m onesAddedMat) At(i, j int) float64 {
	if j == 0 {
		return 1.
	}
	return m.X.At(i, j-1)
}
func (m onesAddedMat) T() mat.Matrix { panic("onesAddedMat T not implemented") }
