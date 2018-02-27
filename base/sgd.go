package base

import (
	//	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// SGDSolver is struct for SGD solver v https://en.wikipedia.org/wiki/Stochastic_gradient_descent
type SGDSolver struct {
	// alpha, StepSize
	StepSize, Momentum         float64
	Adagrad, Adadelta, RMSProp bool
	// running Parameters
	Gt                        mat.Matrix
	Theta, PrevUpdate, Update *mat.Dense
	AdagradG, AdadeltaU       *mat.Dense
	TimeStep, RMSPropGamma    float64
}

// NewSGDSolver returns an initialized sgd solver with stepsize 1e-4 and omentum 0.9
func NewSGDSolver() *SGDSolver {
	s := &SGDSolver{StepSize: 1e-4, Momentum: .9, RMSPropGamma: .9}

	return s
}

// SetTheta should be called before first call to UpdateParams to let the solver know the theta pointer
func (s *SGDSolver) SetTheta(Theta *mat.Dense) { s.Theta = Theta }

// GetTheta can be called anytime after SetTheta to get read access to theta
func (s *SGDSolver) GetTheta() *mat.Dense { return s.Theta }

// GetTimeStep return the number of theta updates already occured
func (s *SGDSolver) GetTimeStep() uint64 { return uint64(s.TimeStep) }

// UpdateParams updates theta from gradient. first call allocates required temporary storage
func (s *SGDSolver) UpdateParams(grad mat.Matrix) {

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
