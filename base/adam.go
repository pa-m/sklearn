package base

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Optimizer has updateParams method to update theta from gradient
type Optimizer interface {
	UpdateParams(grad mat.Matrix)
	SetTheta(Theta *mat.Dense)
	GetTheta() *mat.Dense
	GetTimeStep() uint64
}

// AdamSolver is struct for adam solver v https://arxiv.org/pdf/1412.6980v9.pdf?
type AdamSolver struct {
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

// NewAdamSolver returns an initialized adam solver
func NewAdamSolver() *AdamSolver {
	s := &AdamSolver{StepSize: 1e-3, Beta1: .9, Beta2: .999, Epsilon: 1e-8}

	return s
}

// SetTheta should be called before first call to UpdateParams to let the solver know the theta pointer
func (s *AdamSolver) SetTheta(Theta *mat.Dense) { s.Theta = Theta }

// GetTheta can be called anytime after SetTheta to get read access to theta
func (s *AdamSolver) GetTheta() *mat.Dense { return s.Theta }

// GetTimeStep return the number of theta updates already occured
func (s *AdamSolver) GetTimeStep() uint64 { return uint64(s.TimeStep) }

// UpdateParams updates theta from gradient. first call allocates required temporary storage
func (s *AdamSolver) UpdateParams(grad mat.Matrix) {

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

// -----------
