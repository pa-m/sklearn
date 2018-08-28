package neuralnetwork

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type activationStruct struct{}

// ActivationFunctions WIP
type ActivationFunctions interface {
	Func(z, h *mat.Dense)
	Grad(z, h, grad *mat.Dense)
}
type identityActivation struct{ activationStruct }

func (identityActivation) Func(z, h *mat.Dense) { h.Copy(z) }
func (identityActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. })
}

type logisticActivation struct{ activationStruct }

func (logisticActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 { return 1. / (1. + math.Exp(-z)) })
	//h.Copy(matApply{Matrix: z, Func: func(z float64) float64 { return 1. / (1. + math.Exp(-z)) }})
}
func (logisticActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return h * (1. - h) })
}

type tanhActivation struct{ activationStruct }

func (tanhActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, math.Tanh)

}
func (tanhActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. - h*h })
}

type reluActivation struct{ activationStruct }

func (reluActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 { return math.Max(0, z) })
}
func (reluActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h <= 0 {
			return 0.
		}
		return 1.
	})

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
