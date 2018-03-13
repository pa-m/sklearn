package neuralNetwork

import (
	"fmt"
	"math"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

type lossBaseStruct struct{}

// LossFunctions is the interface for matLoss (matSquareLoss,...)
type LossFunctions interface {
	Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64
}

type squareLoss struct{ lossBaseStruct }

func (squareLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	Ydiff := matSub{A: Ypred, B: Ytrue}
	// J:=(h-y)^2/2
	J := metrics.MeanSquaredError(Ytrue, Ypred, nil, "").At(0, 0)
	// Grad:=(h-y)
	if Grad != nil {
		Grad.Scale(1./float64(nSamples), Ydiff)
	}
	return J
}

type logLoss struct{ lossBaseStruct }

func (logLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	// J:=-y log(h)
	J := -mat.Sum(matMulElem{A: Ytrue, B: base.MatApply1{Matrix: Ypred, Func: math.Log}}) / float64(nSamples)
	// Grad:=-y/h
	if Grad != nil {
		Gfun := func(y, h float64) float64 { return -y / h }
		Grad.Scale(1./float64(nSamples), matApply2{A: Ytrue, B: Ypred, Func: Gfun})
	}
	return J
}

type crossEntropyLoss struct{ lossBaseStruct }

func (crossEntropyLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	nSamples, _ := Ytrue.Dims()
	// J:=-y log(h)-(1-y) log(1-h)
	Jfun := func(y, h float64) float64 {
		eps := 1e-15
		if h <= 0 {
			h = eps
		} else if h >= 1. {
			h = 1 - eps
		}
		return -y*math.Log(h) - (1.-y)*math.Log(1.-h)
	}
	//fmt.Printf("h11=%f J11=%f\n", Ypred.At(0, 0), Jfun(Ytrue.At(0, 0), Ypred.At(0, 0))/float64(nSamples))
	J := mat.Sum(matApply2{A: Ytrue, B: Ypred, Func: Jfun}) / float64(nSamples)
	if Grad != nil {
		// Grad:=-y/h+(1-y)/(1-h)
		Gfun := func(y, h float64) float64 {
			eps := 1e-12
			if h <= 0 {
				h = eps
			} else if h >= 1. {
				h = 1 - eps
			}
			return -y/h + (1-y)/(1-h)
		}
		Grad.Scale(1./float64(nSamples), matApply2{A: Ytrue, B: Ypred, Func: Gfun})
	}
	return J
}

// SupportedLoss are the map[string]Losser of available matrix loss function providers
var SupportedLoss = map[string]LossFunctions{
	"square":        squareLoss{},
	"log":           logLoss{},
	"cross-entropy": crossEntropyLoss{},
}

// NewLoss creates a LossFunctions by its name
func NewLoss(name string) LossFunctions {
	loss, ok := SupportedLoss[name]
	if !ok {
		panic(fmt.Errorf("loss %s is unknown", name))
	}
	return loss
}

type activationStruct struct{}

// ActivationFunctions WIP
type ActivationFunctions interface {
	Func(z, h *mat.Dense)
	Grad(z, h mat.Matrix, grad *mat.Dense)
}
type identityActivation struct{ activationStruct }

func (identityActivation) Func(z, h *mat.Dense) { h.Copy(z) }
func (identityActivation) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{Matrix: h, Func: func(h float64) float64 { return 1. }})
}

type logisticActivation struct{ activationStruct }

func (logisticActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 { return 1. / (1. + math.Exp(-z)) })
	//h.Copy(matApply{Matrix: z, Func: func(z float64) float64 { return 1. / (1. + math.Exp(-z)) }})
}
func (logisticActivation) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{Matrix: h, Func: func(h float64) float64 { return h * (1. - h) }})
}

type tanhActivation struct{ activationStruct }

func (tanhActivation) Func(z, h *mat.Dense) {
	h.Copy(matApply{Matrix: z, Func: math.Tanh})
}
func (tanhActivation) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{Matrix: h, Func: func(h float64) float64 { return 1. - h*h }})
}

type reluActivation struct{ activationStruct }

func (reluActivation) Func(z, h *mat.Dense) {
	h.Copy(matApply{Matrix: z, Func: func(z float64) float64 { return math.Max(0, z) }})
}
func (reluActivation) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{Matrix: h, Func: func(h float64) float64 {
		if h <= 0 {
			return 0.
		}
		return 1.
	}})
}

// SupportedActivations is a map[Sing]ActivationFunctions for the supproted activation functions (identity,logistic,tanh,relu)
var SupportedActivations = map[string]ActivationFunctions{
	"identity": identityActivation{},
	"logistic": logisticActivation{},
	"tanh":     tanhActivation{},
	"relu":     reluActivation{},
}

// NewActivation return ActivationFunctions (Func and Grad) from its name (identity,logistic,tanh,relu)
func NewActivation(name string) ActivationFunctions {
	activation, ok := SupportedActivations[name]
	if !ok {
		panic(fmt.Errorf("unknown activation %s", name))
	}
	return activation
}
