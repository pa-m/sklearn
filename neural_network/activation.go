package neuralnetwork

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type activationStruct struct{}

// ActivationFunctions is interface with Func and Grad for activation functions
type ActivationFunctions interface {
	Func(z, h *mat.Dense)
	Grad(z, h, grad *mat.Dense)
	String() string
}

// IdentityActivation implements ActivationFunctions for identity
type IdentityActivation struct{ activationStruct }

// Func for IdentityActivation
func (*IdentityActivation) Func(z, h *mat.Dense) { h.Copy(z) }

// Grad for IndentityActivation
func (*IdentityActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. })
}

// String for identity
func (*IdentityActivation) String() string { return "identity" }

// LogisticActivation implements ActivationFunctions for sigmoid
type LogisticActivation struct{ activationStruct }

// Func for LogisticActivation
func (*LogisticActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 { return 1. / (1. + math.Exp(-z)) })
	//h.Copy(matApply{Matrix: z, Func: func(z float64) float64 { return 1. / (1. + math.Exp(-z)) }})
}

// Grad for LogisticActivation
func (*LogisticActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return h * (1. - h) })
}

// String for logistic
func (*LogisticActivation) String() string { return "logistic" }

// TanhActivation implements ActivationFunctions for Tanh
type TanhActivation struct{ activationStruct }

// Func for tanh
func (*TanhActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, math.Tanh)

}

// Grad for tanh
func (*TanhActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 { return 1. - h*h })
}

// String for tanh
func (*TanhActivation) String() string { return "tanh" }

// ReluActivation implements ActivationFunctions for rectified linear unit
type ReluActivation struct{ activationStruct }

// Func for relu
func (*ReluActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 {
		if z >= 0 {
			return z
		}
		return 0
	})
}

// Grad for relu
func (*ReluActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h <= 0 {
			return 0.
		}
		return 1.
	})

}

// String for relu
func (*ReluActivation) String() string { return "relu" }

// ParamreluActivation implements ActivationFunctions for parametric relu
type ParamreluActivation struct {
	activationStruct
	Param float64
}

// Func for paramrelu
func (act *ParamreluActivation) Func(z, h *mat.Dense) {
	matx{Dense: h}.CopyApplied(z, func(z float64) float64 {
		if z >= 0 {
			return z
		}
		return z * act.Param
	})
}

// Grad for paramrelu
func (act *ParamreluActivation) Grad(z, h, grad *mat.Dense) {
	matx{Dense: grad}.CopyApplied(h, func(h float64) float64 {
		if h >= 0 {
			return 1.
		}
		return act.Param
	})

}

// String for paramrelu
func (act *ParamreluActivation) String() string { return fmt.Sprintf("paramrelu(%g)", act.Param) }

// SetParameter for paramrelu
func (act *ParamreluActivation) SetParameter(v float64) { act.Param = v }

// DefaultParameter for paramrelu
func (act *ParamreluActivation) DefaultParameter() float64 { return 0.01 }

// EluActivation implements ActivationFunctions for exponential linear unit
type EluActivation struct {
	activationStruct
	Param float64
}

// Func for elu
func (act *EluActivation) Func(z, h *mat.Dense) {
	elu := func(x float64) float64 {
		if x >= 0 {
			return x
		}
		return act.Param * (math.Exp(x) - 1)
	}
	matx{Dense: h}.CopyApplied(z, elu)
}

// Grad for elu
func (act *EluActivation) Grad(z, h, grad *mat.Dense) {
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

// String for relu
func (act *EluActivation) String() string { return fmt.Sprintf("elu(%g)", act.Param) }

// SetParameter for elu
func (act *EluActivation) SetParameter(v float64) { act.Param = v }

// DefaultParameter for elu
func (act *ReluActivation) DefaultParameter() float64 { return 0.01 }

type setparameterer interface {
	SetParameter(v float64)
}
type defaultparameterer interface {
	DefaultParameter() float64
}

// SupportedActivations is a map[Sing]ActivationFunctions for the supproted activation functions (identity,logistic,tanh,relu)
var SupportedActivations = map[string]ActivationFunctions{
	"identity":  &IdentityActivation{},
	"logistic":  &LogisticActivation{},
	"tanh":      &TanhActivation{},
	"relu":      &ReluActivation{},
	"elu":       &EluActivation{},
	"paramrelu": &ParamreluActivation{},
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
