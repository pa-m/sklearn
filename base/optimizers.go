package base

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Optimizer has updateParams method to update theta from gradient
type Optimizer interface {
	UpdateParams(grad mat.Matrix)
	SetTheta(Theta *mat.Dense)
	GetTheta() *mat.Dense
	GetTimeStep() uint64
	String() string
}

// SGDOptimizer is struct for SGD solver v https://en.wikipedia.org/wiki/Stochastic_gradient_descent
type SGDOptimizer struct {
	// alpha, StepSize
	StepSize, Momentum         float64
	Adagrad, Adadelta, RMSProp bool
	// running Parameters
	Gt                        mat.Matrix
	Theta, PrevUpdate, Update *mat.Dense
	AdagradG, AdadeltaU       *mat.Dense
	TimeStep, RMSPropGamma    float64
}

// NewSGDOptimizer returns an initialized *SGDOptimizer with stepsize 1e-4 and momentum 0.9
func NewSGDOptimizer() *SGDOptimizer {
	s := &SGDOptimizer{StepSize: 1e-4, Momentum: .9, RMSPropGamma: .9}

	return s
}

// NewAdagradOptimizer return a *SGDOptimizer setup for adagrad
func NewAdagradOptimizer() *SGDOptimizer {
	s := NewSGDOptimizer()
	s.Adagrad = true
	return s
}

// NewAdadeltaOptimizer return a *SGDOptimizer setup for adadelta
func NewAdadeltaOptimizer() *SGDOptimizer {
	s := NewSGDOptimizer()
	s.Adadelta = true
	return s
}

// NewRMSPropOptimizer return a *SGDOptimizer setup for rmsprop
func NewRMSPropOptimizer() *SGDOptimizer {
	s := NewSGDOptimizer()
	s.RMSProp = true
	return s
}

func (s *SGDOptimizer) String() string {
	switch {
	case s.Adagrad:
		return "adagrad"
	case s.RMSProp:
		return "rmsprop" + fmt.Sprintf(" gamma:%g", s.RMSPropGamma)
	case s.Adadelta:
		return "adadelta" + fmt.Sprintf(" gamma:%g", s.RMSPropGamma)
	default:
		return "sgd" + fmt.Sprintf(" StepSize:%g,Momentum:%g", s.StepSize, s.Momentum)
	}

}

// NewOptimizer only accepts SGD|adagrad|adadelta|rmsprop|adam
func NewOptimizer(name string) Optimizer {
	switch name {
	case "sgd":
		return NewSGDOptimizer()
	case "adagrad":
		return NewAdagradOptimizer()
	case "adadelta":
		return NewAdadeltaOptimizer()
	case "rmsprop":
		return NewRMSPropOptimizer()
	case "adam":
		return NewAdamOptimizer()
	default:
		panic("NewOptimizer only accepts SGD|adagrad|adadelta|rmsprop|adam")
	}
}

// SetTheta should be called before first call to UpdateParams to let the solver know the theta pointer
func (s *SGDOptimizer) SetTheta(Theta *mat.Dense) { s.Theta = Theta }

// GetTheta can be called anytime after SetTheta to get read access to theta
func (s *SGDOptimizer) GetTheta() *mat.Dense { return s.Theta }

// GetTimeStep return the number of theta updates already occurred
func (s *SGDOptimizer) GetTimeStep() uint64 { return uint64(s.TimeStep) }

// UpdateParams updates theta from gradient. first call allocates required temporary storage
func (s *SGDOptimizer) UpdateParams(grad mat.Matrix) {

	NFeatures, NOutputs := s.Theta.Dims()
	if s.TimeStep == 0. {
		init := func(m *mat.Dense, v0 float64) *mat.Dense {
			m.Apply(func(i int, j int, v float64) float64 { return v0 }, m)
			return m
		}
		s.Update = mat.NewDense(NFeatures, NOutputs, nil)
		s.PrevUpdate = mat.NewDense(NFeatures, NOutputs, nil)
		s.AdagradG = init(mat.NewDense(NFeatures, NOutputs, nil), 1.)
		s.AdadeltaU = init(mat.NewDense(NFeatures, NOutputs, nil), 1.)

	}
	s.TimeStep += 1.
	// gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
	s.Gt = grad

	eta := s.StepSize * 100. / (100. + s.TimeStep)
	if s.RMSProp {
		L2 := 1.

		s.Update.Apply(func(j, o int, v float64) float64 {
			etajo := s.StepSize / L2
			if s.TimeStep > 1 {
				etajo /= math.Sqrt(s.AdagradG.At(j, o))
			}
			gradjo := grad.At(j, o) / L2
			return -etajo * gradjo
		}, s.AdagradG)
		s.AdagradG.Apply(func(j, o int, v float64) float64 {
			gradjo := grad.At(j, o) / L2
			v = v*s.RMSPropGamma + (1.-s.RMSPropGamma)*gradjo*gradjo
			return v
		}, s.AdagradG)
	} else if s.Adagrad {
		L2 := 1.

		// TODO make adagrad work
		s.Update.Apply(func(j, o int, v float64) float64 {
			etajo := s.StepSize / L2
			Gjo := s.AdagradG.At(j, o)
			gradjo := grad.At(j, o) / L2
			if s.TimeStep > 1 {
				etajo /= math.Sqrt(Gjo)
			}
			return -etajo * gradjo
		}, grad)
		// accumulate gradients
		s.AdagradG.Apply(func(j, o int, v float64) float64 {
			gradjo := grad.At(j, o) / L2
			v += gradjo * gradjo
			return v
		}, s.AdagradG)
	} else if s.Adadelta {
		// https://arxiv.org/pdf/1212.5701.pdf
		// accumulate gradients
		s.AdagradG.Apply(func(j, o int, v float64) float64 {
			gradjo := grad.At(j, o)
			return s.RMSPropGamma*v + (1.-s.RMSPropGamma)*gradjo*gradjo
		}, s.AdagradG)
		// compute update
		s.Update.Apply(func(j, o int, gradjo float64) float64 {
			etajo := eta
			if s.TimeStep > 1 {
				etajo = math.Sqrt(s.AdadeltaU.At(j, o)) / math.Sqrt(s.AdagradG.At(j, o))
			}
			return -etajo * gradjo
		}, grad)
		s.AdadeltaU.Apply(func(j, o int, v float64) float64 {
			upd := s.Update.At(j, o)
			return s.RMSPropGamma*v + (1.-s.RMSPropGamma)*upd*upd
		}, s.AdadeltaU)
	} else {
		// normal SGD with momentum
		s.Update.Apply(func(j, o int, v float64) float64 {
			return s.Momentum*s.PrevUpdate.At(j, o) - eta*grad.At(j, o)
		}, grad)
	}
	s.Theta.Add(s.Theta, s.Update)
	if math.IsNaN(s.Theta.At(0, 0)) {
		panic("Nan")
	}
	s.PrevUpdate.Clone(s.Update)
}

// --------

// AdamOptimizer is struct for adam solver v https://arxiv.org/pdf/1412.6980v9.pdf?
type AdamOptimizer struct {
	// alpha, StepSize
	StepSize float64
	// Beta1,Beta2 are exponential decay rate for the moment estimates
	Beta1, Beta2 float64
	// epsilon=1e-8 avoid zero division
	Epsilon float64
	// running Parameters
	Gt                                            mat.Matrix
	Theta, Mt, Vt, Mtcap, Vtcap, Tmp0, Tmp1, Tmp2 *mat.Dense
	TimeStep                                      float64
}

// NewAdamOptimizer returns an initialized adam solver
func NewAdamOptimizer() *AdamOptimizer {
	s := &AdamOptimizer{StepSize: 1e-3, Beta1: .9, Beta2: .999, Epsilon: 1e-8}

	return s
}

func (s *AdamOptimizer) String() string {
	return fmt.Sprintf("adam stepsize:%g beta1:%g beta2:%g epsilon:%g", s.StepSize, s.Beta1, s.Beta2, s.Epsilon)
}

// SetTheta should be called before first call to UpdateParams to let the solver know the theta pointer
func (s *AdamOptimizer) SetTheta(Theta *mat.Dense) { s.Theta = Theta }

// GetTheta can be called anytime after SetTheta to get read access to theta
func (s *AdamOptimizer) GetTheta() *mat.Dense { return s.Theta }

// GetTimeStep return the number of theta updates already occurred
func (s *AdamOptimizer) GetTimeStep() uint64 { return uint64(s.TimeStep) }

// UpdateParams updates theta from gradient. first call allocates required temporary storage
func (s *AdamOptimizer) UpdateParams(grad mat.Matrix) {

	NFeatures, NOutputs := s.Theta.Dims()
	if s.TimeStep == 0. {
		s.Tmp0 = mat.NewDense(NFeatures, NOutputs, nil)
		s.Tmp1 = mat.NewDense(NFeatures, NOutputs, nil)
		s.Tmp2 = mat.NewDense(NFeatures, NOutputs, nil)
		s.Mt = mat.NewDense(NFeatures, NOutputs, nil)
		s.Vt = mat.NewDense(NFeatures, NOutputs, nil)
		s.Mtcap = mat.NewDense(NFeatures, NOutputs, nil)
		s.Vtcap = mat.NewDense(NFeatures, NOutputs, nil)

	}
	s.TimeStep += 1.
	// gt ← ∇θft(θt−1) (Get gradients w.r.t. stochastic objective at timestep t)
	s.Gt = grad
	// mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
	s.Tmp1.Scale(s.Beta1, s.Mt)
	s.Tmp2.Scale(1.-s.Beta1, s.Gt)
	s.Mt.Add(s.Tmp1, s.Tmp2)
	// vt ← β2 · vt−1 + (1 − β2) · gt² (Update biased second raw moment estimate)
	s.Tmp1.Scale(s.Beta2, s.Vt)
	s.Tmp0.MulElem(s.Gt, s.Gt)
	s.Tmp2.Scale(1.-s.Beta2, s.Tmp0)
	s.Vt.Add(s.Tmp1, s.Tmp2)
	// mb t ← mt/(1 − β1^t) (Compute bias-corrected first moment estimate)
	s.Mtcap.Scale(1./(1.-math.Pow(s.Beta1, s.TimeStep)), s.Mt)
	// vbt ← vt/(1 − β2^t) (Compute bias-corrected second raw moment estimate)
	s.Vtcap.Scale(1./(1.-math.Pow(s.Beta2, s.TimeStep)), s.Vt)
	// θt ← θt−1 − α · mb t/(√vbt + epsilon) (Update parameters)
	fn := func(i, j int, Mtcapij float64) float64 {
		return -s.StepSize * Mtcapij / (math.Sqrt(s.Vtcap.At(i, j)) + s.Epsilon)
	}
	s.Tmp2.Apply(fn, s.Mtcap)

	s.Theta.Add(s.Theta, s.Tmp2)
}
