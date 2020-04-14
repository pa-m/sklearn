package gaussianprocess

import (
	"fmt"
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/gaussian_process/kernels"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
	//	"gorgonia.org/tensor"
	"github.com/pa-m/sklearn/gaussian_process/tensor"
	"math"
)

// Regressor ...
type Regressor struct {
	kernels.Kernel
	Alpha []float64
	// Optimizer is always optimize.LBFGS
	NRestartsOptimizer int
	NormalizeY         bool
	// copy_X_train is always true
	base.RandomState
	Xtrain                     *mat.Dense
	Ytrain                     *mat.Dense
	YtrainMean                 *mat.Dense
	KernelOpt                  kernels.Kernel
	L                          *mat.Cholesky
	LogMarginalLikelihoodValue float64
}

// NewRegressor ...
func NewRegressor(kernel kernels.Kernel) *Regressor {
	gp := &Regressor{
		Kernel: kernel,
		Alpha:  []float64{1e-10},
	}
	gp.KernelOpt = gp.Kernel
	return gp
}

// IsClassifier returns false
func (m *Regressor) IsClassifier() bool { return false }

// PredicterClone clones Predicter (for KFold...)
func (m *Regressor) PredicterClone() base.Predicter {
	clone := *m
	if cloner, ok := m.RandomState.(base.SourceCloner); ok {
		clone.RandomState = cloner.SourceClone()
	}
	if m.Xtrain != nil {
		clone.Xtrain = mat.DenseCopyOf(m.Xtrain)

		clone.Ytrain = mat.DenseCopyOf(m.Ytrain)
	}
	clone.Kernel = m.Kernel.CloneWithTheta(m.Kernel.Theta())
	if m.KernelOpt != kernels.Kernel(nil) {
		clone.KernelOpt = m.KernelOpt.CloneWithTheta(m.KernelOpt.Theta())
	}
	if m.L != nil {
		clone.L.Clone(m.L)
	}
	return &clone
}

// GetNOutputs returns Y columns count
func (m *Regressor) GetNOutputs() int {
	if m.Ytrain == nil {
		return 1
	}
	return m.Ytrain.RawMatrix().Cols
}

// Fit Gaussian process regression model
func (m *Regressor) Fit(X, Y mat.Matrix) base.Fiter {
	_, ry := Y.Dims()
	m.Xtrain = mat.DenseCopyOf(X)
	m.Ytrain = mat.DenseCopyOf(Y)
	if len(m.Alpha) != 1 && len(m.Alpha) != ry {
		panic(fmt.Errorf("alpha must be a scalar or an array with same number of entries as y.(%d != %d)", len(m.Alpha), ry))
	}
	return m
}

// PredictEx predicts using the Gaussian process regression model, returning Ymean and std or cov
func (m *Regressor) PredictEx(X mat.Matrix, Y mat.Mutable, returnStd, returnCov bool) (*mat.Dense, *mat.DiagDense, *mat.Dense) {
	NSamples, _ := X.Dims()
	var Ymean, Ycov *mat.Dense
	var Ystd *mat.DiagDense
	if _, ok := Y.(*mat.Dense); ok {
		Ymean = Y.(*mat.Dense)
	} else {
		Ymean = mat.NewDense(NSamples, m.GetNOutputs(), nil)
	}
	if m.Xtrain == nil {
		// # Unfitted;predict based on GP prior
		// y_mean = np.zeros(X.shape[0])
		if returnCov {
			Ycov, _ = m.Kernel.Eval(X, nil, false)
		} else if returnStd {
			Ystd = m.Kernel.Diag(X)
			for i := 0; i < NSamples; i++ {
				Ystd.SetDiag(i, math.Sqrt(Ystd.At(i, i)))
			}
		}

	} else {
		// # Predict based on GP posterior
		//K_trans = self.kernel_(X, self.X_train_)
		//y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
		//y_mean = self._y_train_mean + y_mean  # undo normal.

		Ktrans, _ := m.Kernel.Eval(X, m.Xtrain, false)
		Ymean.Mul(Ktrans, mat.NewDense(NSamples, 1, m.Alpha))
		Ymean.Add(Ymean, m.YtrainMean)
		if returnStd {
			// TODO
		} else if returnCov {
			// TODO
		}
	}
	return base.FromDense(Y, Ymean), Ystd, Ycov
}

// Predict using the Gaussian process regression model
func (m *Regressor) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	Ymean, _, _ := m.PredictEx(X, Y, false, false)
	return base.FromDense(Y, Ymean)
}

// Score returns R2 score
func (m *Regressor) Score(X, Y mat.Matrix) float64 {
	m.Fit(X, Y)
	pred := m.Predict(X, nil)
	return metrics.R2Score(Y, pred, nil, "").At(0, 0)
}

// LogMarginalLikelihood returns log-marginal likelihood of theta for training data
func (m *Regressor) LogMarginalLikelihood(Theta mat.Matrix, evalGradient bool) (
	lml float64, grad []float64,
) {
	if Theta == mat.Matrix(nil) {
		if evalGradient {
			panic("Gradient can only be evaluated for theta!=nil")
		}
		return m.LogMarginalLikelihoodValue, nil
	}
	kernel := m.KernelOpt.CloneWithTheta(Theta)
	var K *mat.Dense
	var Kg *tensor.Dense
	_ = Kg

	K, Kg = kernel.Eval(m.Xtrain, nil, evalGradient)

	// add m.Alpha to K diagonal
	Kraw := K.RawMatrix()
	Kdata := Kraw.Data
	alphainc := 1
	if len(m.Alpha) == 1 {
		alphainc = 0
	}
	nx, _ := m.Xtrain.Dims()
	for i := 0; i < nx; i++ {
		Kdata[i*(nx+1)] += m.Alpha[i*alphainc]
	}
	Ksym := &mat.SymDense{}
	Ksym.SetRawSymmetric(blas64.Symmetric{Uplo: blas.Upper, N: Kraw.Rows, Stride: Kraw.Stride, Data: Kraw.Data})
	cho := &mat.Cholesky{}

	posdef := cho.Factorize(Ksym)
	thetarows, _ := Theta.Dims()
	if evalGradient {
		grad = make([]float64, thetarows)
	}
	if !posdef {
		return math.Inf(-1), grad
	}
	// # Support multi-dimensional output of self.y_train_
	// L = cholesky(K, lower=True)  # Line 2
	L := &mat.TriDense{}
	cho.LTo(L)
	// alpha = cho_solve((L, True), y_train)  # Line 3
	alpha := &mat.Dense{}
	err := cho.SolveTo(alpha, m.Ytrain)

	if err != nil {
		panic(err)
	}
	// log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
	// log_likelihood_dims -= np.log(np.diag(L)).sum()
	// log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
	ry, cy := m.Ytrain.Dims()
	lldims := mat.NewVecDense(cy, nil)
	tmp := mat.NewVecDense(cy, nil)
	for i := 0; i < ry; i++ {
		tmp.MulElemVec(m.Ytrain.RowView(i), alpha.RowView(i))
		lldims.AddVec(lldims, tmp)

	}
	sub1 := 0.
	for i := 0; i < ry; i++ {
		sub1 += math.Log(L.At(i, i))
	}
	sub2 := float64(nx) / 2 * math.Log(2*math.Pi)
	for i := 0; i < cy; i++ {
		lldims.SetVec(i, -.5*lldims.AtVec(i)-sub1-sub2)
	}

	// log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
	lml = mat.Sum(lldims)
	//if eval_gradient:  # compare Equation 5.9 from GPML
	if evalGradient {
		// tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
		// tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
		// # Compute "0.5 * trace(tmp.dot(K_gradient))" without
		// # constructing the full matrix tmp.dot(K_gradient) since only
		// # its diagonal is required
		// log_likelihood_gradient_dims = \
		// 0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
		// log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
		choSolved := &mat.Dense{}
		err = cho.SolveTo(choSolved, eye(nx))
		if err != nil {
			panic(err)
		}
		ni, nk := alpha.Dims()
		nj := ni
		tmp := tensor.NewDense(tensor.Float64, tensor.Shape{ni, nj, nk})
		tmpdata := tmp.Data().([]float64)
		alpraw := alpha.RawMatrix()
		alpdata := alpraw.Data
		for i, tmpipos, a1pos := 0, 0, 0; i < ni; i, tmpipos, a1pos = i+1, tmpipos+nj*nk, a1pos+alpraw.Stride {

			for j, tmpjpos, a2pos := 0, tmpipos, 0; j < nj; j, tmpjpos, a2pos = j+1, tmpjpos+nk, a2pos+alpraw.Stride {
				for k := 0; k < nk; k++ {
					tmpdata[tmpjpos+k] = alpdata[a1pos+k]*alpdata[a2pos+k] - choSolved.At(i, j)
				}
			}
		}
		//# Compute "0.5 * trace(tmp.dot(K_gradient))" without
		//# constructing the full matrix tmp.dot(K_gradient) since only
		//# its diagonal is required
		//log_likelihood_gradient_dims = \
		//0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
		tmpShape, tmpStrides := tmp.Shape(), tmp.Strides()
		KgShape, KgStrides := Kg.Shape(), Kg.Strides()
		llgdims := mat.NewDense(KgShape[2], tmpShape[2], nil)
		Kgd := Kg.Data().([]float64)
		{
			llgd := llgdims.RawMatrix()
			for k := 0; k < llgd.Rows; k++ {
				for l := 0; l < llgd.Cols; l++ {
					sum := 0.
					for i, tmpil, Kgik := 0, l, k; i < tmpShape[0]; i, tmpil, Kgik = i+1, tmpil+tmpStrides[0], Kgik+KgStrides[0] {
						for j, tmpijl, Kgijk := 0, tmpil, Kgik; j < tmpShape[1]; j, tmpijl, Kgijk = j+1, tmpijl+tmpStrides[1], Kgijk+KgStrides[1] {
							sum += tmpdata[tmpijl] * Kgd[Kgijk]
						}
					}
					llgdims.Set(k, l, .5*sum)
				}
			}
		}
		grad = make([]float64, KgShape[2])
		for i := range grad {
			grad[i] = mat.Sum(llgdims.RowView(i))

		}
	}
	return
}
