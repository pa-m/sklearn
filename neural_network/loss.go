package neuralNetwork

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type matLoss struct{}

// Losser is the interface for matLoss (matSquareLoss,...)
type Losser interface {
	Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64
}

type matSquareLoss struct{ matLoss }

func (matSquareLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	// J:=(h-y)^2/2
	Ydiff := matSub{Ypred, Ytrue}
	square := func(yd float64) float64 { return yd * yd }
	J := mat.Sum(matApply{Ydiff, square})
	// Grad:=(h-y)
	if Grad != nil {
		Grad.Copy(Ydiff)
	}
	return J
}

type matLogLoss struct{ matLoss }

func (matLogLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	// J:=-y log(h)
	J := -mat.Sum(matMulElem{Ytrue, matApply{Ypred, math.Log}})
	// Grad:=-y/h
	if Grad != nil {
		Gfun := func(y, h float64) float64 { return -y / h }
		Grad.Copy(matApply2{Ytrue, Ypred, Gfun})
	}
	return J
}

type matCrossEntropyLoss struct{ matLoss }

func (matCrossEntropyLoss) Loss(Ytrue, Ypred mat.Matrix, Grad *mat.Dense) float64 {
	// J:=-y log(h)-(1-y) log(1-h)
	Jfun := func(y, h float64) float64 {
		eps := 1e-12
		if h <= 0 {
			h = eps
		} else if h >= 1. {
			h = 1 - eps
		}
		return -y*math.Log(h) - (1.-y)*math.Log(1.-h)
	}
	J := mat.Sum(matApply2{Ytrue, Ypred, Jfun})
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
		Grad.Copy(matApply2{Ytrue, Ypred, Gfun})
	}
	return J
}

// LossWrtPred are the map[string]Losser of available matrix loss function providers
var LossWrtPred = map[string]Losser{
	"square":        matSquareLoss{},
	"log":           matLogLoss{},
	"cross-entropy": matCrossEntropyLoss{},
}

type activation2 struct{}

// Activation2Grader WIP
type Activation2Grader interface {
	Func(z mat.Matrix, h *mat.Dense)
	Grad(z, h mat.Matrix, grad *mat.Dense)
}
type actIdentity struct{ activation2 }

func (actIdentity) Func(z mat.Matrix, h *mat.Dense) { h.Copy(z) }
func (actIdentity) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{h, func(h float64) float64 { return 1. }})
}

type actLogistic struct{ activation2 }

func (actLogistic) Func(z mat.Matrix, h *mat.Dense) {
	h.Copy(matApply{z, func(z float64) float64 { return 1. / (1. + math.Exp(-z)) }})
}
func (actLogistic) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{h, func(h float64) float64 { return h * (1. - h) }})
}

type actTanh struct{ activation2 }

func (actTanh) Func(z mat.Matrix, h *mat.Dense) {
	h.Copy(matApply{z, math.Tanh})
}
func (actTanh) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{h, func(h float64) float64 { return 1. - h*h }})
}

type actReLU struct{ activation2 }

func (actReLU) Func(z mat.Matrix, h *mat.Dense) {
	h.Copy(matApply{z, func(z float64) float64 { return math.Max(0, z) }})
}
func (actReLU) Grad(z, h mat.Matrix, grad *mat.Dense) {
	grad.Copy(matApply{h, func(h float64) float64 {
		if h <= 0 {
			return 0.
		}
		return 1.
	}})
}

var matActivations = map[string]Activation2Grader{
	"identity": actIdentity{},
	"logistic": actLogistic{},
	"tanh":     actTanh{},
	"relu":     actReLU{},
}
