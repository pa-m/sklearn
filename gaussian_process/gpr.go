package gaussianprocess

import (
	"fmt"
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/gaussian_process/kernels"
	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	"math"
)

// GaussianProcessRegressor ...
type GaussianProcessRegressor struct {
	kernels.Kernel
	Alpha []float64
	// Optimizer is always optimize.LBFGS
	NRestartsOptimizer int
	NormalizeY         bool
	// copy_X_train is always true
	base.RandomState
	Xtrain                     *mat.Dense
	Ytrain                     *mat.Dense
	KernelOpt                  kernels.Kernel
	L                          *mat.Cholesky
	LogMarginalLikelihoodValue float64
}

// NewGaussianProcessRegressor ...
func NewGaussianProcessRegressor(kernel kernels.Kernel) *GaussianProcessRegressor {
	gp := &GaussianProcessRegressor{
		Kernel: kernel,
		Alpha:  []float64{1e-10},
	}
	gp.KernelOpt = gp.Kernel
	return gp
}
func (m *GaussianProcessRegressor) IsClassifier() bool { return false }
func (m *GaussianProcessRegressor) PredicterClone() base.Predicter {
	clone := *m
	if cloner, ok := m.RandomState.(base.SourceCloner); ok {
		clone.RandomState = cloner.Clone()
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
func (m *GaussianProcessRegressor) GetNOutputs() int {
	return m.Ytrain.RawMatrix().Cols
}
func (m *GaussianProcessRegressor) Fit(X, Y mat.Matrix) base.Fiter {
	_, ry := Y.Dims()
	m.Xtrain = mat.DenseCopyOf(X)
	m.Ytrain = mat.DenseCopyOf(Y)
	if len(m.Alpha) != 1 && len(m.Alpha) != ry {
		panic(fmt.Errorf("alpha must be a scalar or an array with same number of entries as y.(%d != %d)", len(m.Alpha), ry))
	}
	return m
}
func (m *GaussianProcessRegressor) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	NSamples, _ := X.Dims()
	var Yd *mat.Dense
	if _, ok := Y.(*mat.Dense); ok {
		Yd = Y.(*mat.Dense)
	} else {
		Yd = mat.NewDense(NSamples, m.GetNOutputs(), nil)
	}
	return base.FromDense(Y, Yd)
}
func (m *GaussianProcessRegressor) Score(X, Y mat.Matrix) float64 {
	m.Fit(X, Y)
	pred := m.Predict(X, nil)
	return metrics.R2Score(Y, pred, nil, "").At(0, 0)
}

// LogMarginalLikelihood returns log-marginal likelihood of theta for training data
func (m *GaussianProcessRegressor) LogMarginalLikelihood(Theta mat.Matrix, evalGradient bool) (
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
		{
			Kgd := Kg.Data().([]float64)
			llgd := llgdims.RawMatrix()
			for k := 0; k < llgd.Rows; k++ {
				for l := 0; l < llgd.Cols; l++ {
					sum := 0.
					for i, tmpposil, Kgposik := 0, l, k; i < tmpShape[0]; i, tmpposil, Kgposik = i+1, tmpposil+tmpStrides[0], Kgposik+KgStrides[0] {
						for j, tmpposijl, Kgposijk := 0, tmpposil, Kgposik; j < tmpShape[1]; j, tmpposijl, Kgposijk = j+1, tmpposijl+tmpStrides[1], Kgposijk+KgStrides[1] {
							sum += tmpdata[tmpposijl] * Kgd[Kgposijk]
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
