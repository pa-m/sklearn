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

func (*identityActivation) Func(z, h *mat.Dense) { h.Copy(z) }
func (*identityActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. })
}

type logisticActivation struct{ activationStruct }

func (*logisticActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 { return 1. / (1. + math.Exp(-z)) })
	//h.Copy(matApply{Matrix: z, Func: func(z float64) float64 { return 1. / (1. + math.Exp(-z)) }})
}
func (*logisticActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return h * (1. - h) })
}

type tanhActivation struct{ activationStruct }

func (*tanhActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, math.Tanh)

}
func (*tanhActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. - h*h })
}

type reluActivation struct{ activationStruct }

func (*reluActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 {
		if z >= 0 {
			return z
		}
		return 0
	})
}
func (*reluActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h <= 0 {
			return 0.
		}
		return 1.
	})

}

type paramreluActivation struct {
	activationStruct
	Param float64
}

func (act *paramreluActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 {
		if z >= 0 {
			return z
		}
		return z * act.Param
	})
}
func (act *paramreluActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h >= 0 {
			return 1.
		}
		return act.Param
	})

}
func (act *paramreluActivation) SetParameter(v float64)    { act.Param = v }
func (act *paramreluActivation) DefaultParameter() float64 { return 0.01 }

type eluActivation struct {
	activationStruct
	Param float64
}

func (act *eluActivation) Func(z, h *mat.Dense) {
	elu := func(x float64) float64 {
		if x >= 0 {
			return x
		}
		return act.Param * (math.Exp(x) - 1)
	}
	matx{Dense: h}.CopyApplied(z, elu)
}

func (act *eluActivation) Grad(z, h, grad *mat.Dense) {
	if act.Param < 0 {
		panic("elu Param must be >0")
	}
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h <= 0 {
			// derivative of a*e^x-a is a*e^x that is h+a
			return h + act.Param
		}
		return 1.
	})

}
func (act *eluActivation) SetParameter(v float64)     { act.Param = v }
func (act *reluActivation) DefaultParameter() float64 { return 0.01 }

type setparameterer interface {
	SetParameter(v float64)
}
type defaultparameterer interface {
	DefaultParameter() float64
}

// SupportedActivations is a map[Sing]ActivationFunctions for the supproted activation functions (identity,logistic,tanh,relu)
var SupportedActivations = map[string]ActivationFunctions{
	"identity":  &identityActivation{},
	"logistic":  &logisticActivation{},
	"tanh":      &tanhActivation{},
	"relu":      &reluActivation{},
	"elu":       &eluActivation{},
	"paramrelu": &paramreluActivation{},
}

// NewActivation return ActivationFunctions (Func and Grad) from its name (identity,logistic,tanh,relu)
// arg may be a string or an ActivationFunctions
func NewActivation(arg interface{}) ActivationFunctions {
	if f, ok := arg.(ActivationFunctions); ok {
		return f
	}
	name := arg.(string)
	activation, ok := SupportedActivations[name]
	if !ok {
		panic(fmt.Errorf("unknown activation %s", name))
	}
	if a1, ok := activation.(defaultparameterer); ok {
		if a2, ok := activation.(setparameterer); ok {
			a2.SetParameter(a1.DefaultParameter())
		}
	}
	return activation
}
