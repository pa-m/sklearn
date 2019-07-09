package kernels

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

// Kernel interface
type Kernel interface {
	Eval(X, Y mat.Matrix) *mat.Dense
	Diag(X mat.Matrix) (K *mat.DiagDense)
	IsStationary() bool
	String() string
}

// StationaryKernelMixin mixin for kernels which are stationary: k(X, Y)= f(X-Y)
type StationaryKernelMixin struct{}

// IsStationary returns whether the kernel is stationary
func (StationaryKernelMixin) IsStationary() bool { return true }

// NormalizedKernelMixin is Mixin for kernels which are normalized: k(X, X)=1
type NormalizedKernelMixin struct{}

// Diag returns the diagonal of the kernel k(X, X)
func (k NormalizedKernelMixin) Diag(X mat.Matrix) (K *mat.DiagDense) {
	nx, _ := X.Dims()

	K = mat.NewDiagDense(nx, nil)
	for ix := 0; ix < nx; ix++ {
		K.SetDiag(ix, 1)

	}
	return
}

// KernelOperator is a kernel based on two others
type KernelOperator struct {
	k1, k2 Kernel
}

// IsStationary returns whether the kernel is stationary
func (k *KernelOperator) IsStationary() bool {
	return k.k1.IsStationary() && k.k2.IsStationary()
}

// Sum kernel k1 + k2 of two kernels k1 and k2
type Sum struct {
	KernelOperator
}
// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Sum) Eval(X, Y mat.Matrix) *mat.Dense {

	K1 := k.k1.Eval(X, Y)
	K2 := k.k2.Eval(X, Y)
	K1.Add(K1, K2)
	return K1
}
// Diag returns the diagonal of the kernel k(X, X)
func (k *Sum) Diag(X mat.Matrix) *mat.DiagDense {
	K1 := k.k1.Diag(X)
	K2 := k.k2.Diag(X)
	nx, _ := K1.Dims()
	for i := 0; i < nx; i++ {
		K1.SetDiag(i, K1.At(i, i)+K2.At(i, i))
	}
	return K1
}
// String ...
func (k *Sum) String() string {
	return k.k1.String() + " + " + k.k2.String()
}
// Product kernel k1 * k2 of two kernels k1 and k2
type Product struct {
	KernelOperator
}
// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Product) Eval(X, Y mat.Matrix) *mat.Dense {

	K1 := k.k1.Eval(X, Y)
	K2 := k.k2.Eval(X, Y)
	K1.MulElem(K1, K2)
	return K1
}
// Diag returns the diagonal of the kernel k(X, X)
func (k *Product) Diag(X mat.Matrix) *mat.DiagDense {
	K1 := k.k1.Diag(X)
	K2 := k.k2.Diag(X)
	nx, _ := K1.Dims()
	for i := 0; i < nx; i++ {
		K1.SetDiag(i, K1.At(i, i)*K2.At(i, i))
	}
	return K1
}
// String ...
func (k *Product) String() string {
	return k.k1.String() + " * " + k.k2.String()
}
// Exponentiation exponentiate kernel by given exponent
type Exponentiation struct {
	Kernel
	Exponent float64
}
// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Exponentiation) Eval(X, Y mat.Matrix) *mat.Dense {
	K := k.Kernel.Eval(X, Y)
	K.Apply(func(_, _ int, v float64) float64 {
		return math.Pow(v, k.Exponent)
	}, K)
	return K
}
// Diag returns the diagonal of the kernel k(X, X)
func (k *Exponentiation) Diag(X mat.Matrix) *mat.DiagDense {
	K := k.Kernel.Diag(X)
	nx, _ := K.Dims()
	for i := 0; i < nx; i++ {
		K.SetDiag(i, math.Pow(K.At(i, i), k.Exponent))
	}
	return K
}
// String ...
func(k* Exponentiation) String() string {
	return fmt.Sprintf("%s ** %g", k.Kernel.String(),k.Exponent)
}

// ConstantKernel can be used as part of a product-kernel where it scales the magnitude of the other factor (kernel) or as part of a sum-kernel, where it modifies the mean of the Gaussian process.
// k(x_1, x_2) = constant_value for all x_1, x_2
type ConstantKernel struct {
	ConstantValue       float64
	ConstantValueBounds [2]float64
	StationaryKernelMixin
}

// Eval returns
// K : array, shape (n_samples_X, n_samples_Y)
// Kernel k(X, Y)
func (k *ConstantKernel) Eval(X, Y mat.Matrix) *mat.Dense {
	nx, _ := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	K := mat.NewDense(nx, ny, nil)
	kdata := K.RawMatrix().Data
	for i := range kdata {
		kdata[i] = k.ConstantValue
	}
	return K
}

// Diag returns the diagonal of the kernel k(X, X).
func (k *ConstantKernel) Diag(X mat.Matrix) (K *mat.DiagDense) {
	nx, _ := X.Dims()

	K = mat.NewDiagDense(nx, nil)
	data := K.RawSymBand().Data
	for i := range data {
		data[i] = k.ConstantValue
	}
	return
}

// String returns  string representation of kernel
func (k *ConstantKernel) String() string {
	return fmt.Sprintf("%.3g**2", math.Sqrt(k.ConstantValue))
}

// WhiteKernel ...
// The main use-case of this kernel is as part of a sum-kernel where it
// explains the noise-component of the signal. Tuning its parameter
// corresponds to estimating the noise-level.
// k(x_1, x_2) = noise_level if x_1 == x_2 else 0
type WhiteKernel struct {
	NoiseLevel       float64
	NoiseLevelBounds [2]float64
	StationaryKernelMixin
}

// Eval return the kernel k(X, Y)
func (k *WhiteKernel) Eval(X, Y mat.Matrix) *mat.Dense {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	K := mat.NewDense(nx, ny, nil)
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)
	for ix := 0; ix < nx; ix++ {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			if floats.EqualApprox(Xrow, Yrow, 1e-8) {
				K.Set(ix, iy, k.NoiseLevel)
			}
		}
	}
	return K
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *WhiteKernel) Diag(X mat.Matrix) *mat.DiagDense {
	nx, _ := X.Dims()
	K := mat.NewDiagDense(nx, nil)
	for ix := 0; ix < nx; ix++ {
		K.SetDiag(ix, k.NoiseLevel)
	}
	return K
}

// String returns  string representation of kernel
func (k *WhiteKernel) String() string {
	return fmt.Sprintf("WhiteKernel(noise_level=%.3g)", k.NoiseLevel)
}

// RBF kernel is a stationary kernel. It is also known as the
// "squared exponential" kernel. It is parameterized by a length-scale
// parameter length_scale>0, which can either be a scalar (isotropic variant
// of the kernel) or a vector with the same number of dimensions as the inputs
// X (anisotropic variant of the kernel). The kernel is given by:
// k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
// This kernel is infinitely differentiable, which implies that GPs with this
// kernel as covariance function have mean square derivatives of all orders,
// and are thus very smooth.
type RBF struct {
	LengthScale       interface{}
	LengthScaleBounds [2]float64
	StationaryKernelMixin
	NormalizedKernelMixin
}

// Eval return the kernel k(X, Y)
func (k *RBF) Eval(X, Y mat.Matrix) (K *mat.Dense) {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	var scale func([]float64, int) float64
	switch lengthScale := k.LengthScale.(type) {
	case int:
		scale = func(X []float64, feat int) float64 { return X[feat] / float64(lengthScale) }
	case float64:
		scale = func(X []float64, feat int) float64 { return X[feat] / lengthScale }
	case []float64:
		if len(lengthScale) != nfeat {
			panic("LengthScale has wrong dimension")
		}
		scale = func(X []float64, feat int) float64 { return X[feat] / lengthScale[feat] }
	default:
		panic("float64 or []float64 expected")
	}

	K = mat.NewDense(nx, ny, nil)
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)

	// K=np.exp(-0.5 * cdist(X/K.length_scale,Y/K.length_scale,'sqeuclidean'))
	for ix := 0; ix < nx; ix++ {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			var d2 float64
			for feat := 0; feat < nfeat; feat++ {
				d := scale(Xrow, feat) - scale(Yrow, feat)
				d2 += d * d
			}
			K.Set(ix, iy, math.Exp(-.5*(d2)))
		}
	}
	return
}

// IsAnisotropic ...
func (k *RBF) IsAnisotropic() bool {
	if s, ok := k.LengthScale.([]float64); ok && len(s) > 1 {
		return true
	}
	return false
}

// String returns  string representation of kernel
func (k *RBF) String() string {
	return fmt.Sprintf("RBF(%v)", k.LengthScale)
}

// DotProduct kernel
// The DotProduct kernel is non-stationary and can be obtained from linear
// regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
// . , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
// is invariant to a rotation of the coordinates about the origin, but not
// translations. It is parameterized by a parameter sigma_0^2. For
// sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
// it is inhomogeneous. The kernel is given by
// k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j
// The DotProduct kernel is commonly combined with exponentiation.
type DotProduct struct {
	Sigma0       float64
	Sigma0Bounds [2]float64
}

// Eval return the kernel k(X, Y)
func (k *DotProduct) Eval(X, Y mat.Matrix) (K *mat.Dense) {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	K = mat.NewDense(nx, ny, nil)
	s2 := k.Sigma0 * k.Sigma0
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)
	for ix := 0; ix < nx; ix++ {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			K.Set(ix, iy, s2+mat.Dot(mat.NewVecDense(nfeat, Xrow), mat.NewVecDense(nfeat, Yrow)))
		}
	}
	return
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *DotProduct) Diag(X mat.Matrix) (K *mat.DiagDense) {
	n, nfeat := X.Dims()
	K = mat.NewDiagDense(n, nil)
	s2 := k.Sigma0 * k.Sigma0
	Xrow := make([]float64, nfeat)
	for i := 0; i < n; i++ {
		mat.Row(Xrow, i, X)
		Vrow := mat.NewVecDense(nfeat, Xrow)
		K.SetDiag(i, s2+mat.Dot(Vrow, Vrow))
	}
	return
}

// IsStationary returns whether the kernel is stationary
func (k *DotProduct) IsStationary() bool { return false }

// String returns  string representation of kernel
func (k *DotProduct) String() string {
	return fmt.Sprintf("DotProduct(sigma_0=%.3g)", k.Sigma0)
}
