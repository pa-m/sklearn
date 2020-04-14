package kernels

import (
	"fmt"
	"github.com/pa-m/sklearn/base"
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
	//t "gorgonia.org/tensor"
	t "github.com/pa-m/sklearn/gaussian_process/tensor"
)

// hyperparameter specification
// for no value_type is mat.Vector
// bounds is [2]float64
// IsFixed() is true if bounds are equal
type hyperparameter struct {
	Name    string
	PValue  *float64
	PBounds *[2]float64
}

// IsFixed return true when bounds are equal or missing
func (param hyperparameter) IsFixed() bool {
	return param.PBounds == nil || (*param.PBounds)[1] == (*param.PBounds)[0]
}

// hyperparameters ...
type hyperparameters []hyperparameter

func (params hyperparameters) notFixed() hyperparameters {
	notFixed := hyperparameters{}
	for _, p := range params {
		if !p.IsFixed() {
			notFixed = append(notFixed, p)
		}
	}
	return notFixed
}
func (params hyperparameters) Theta() mat.Matrix {
	notFixed := params.notFixed()
	return matFromFunc{
		r: len(notFixed),
		c: 1,

		at: func(i, j int) float64 {
			return math.Log(*notFixed[i].PValue)
		},
		set: func(i, j int, v float64) {
			*notFixed[i].PValue = math.Exp(v)
		},
	}
}
func (params hyperparameters) Bounds() (t mat.Matrix) {
	notFixed := params.notFixed()
	return matFromFunc{
		r: len(notFixed),
		c: 2,
		at: func(i, j int) float64 {
			return math.Log((*notFixed[i].PBounds)[j])
		},
		set: func(i, j int, v float64) {
			(*notFixed[i].PBounds)[j] = math.Exp(v)
		},
	}
}

// Kernel interface
type Kernel interface {
	hyperparameters() hyperparameters
	Theta() mat.Matrix
	Bounds() mat.Matrix
	CloneWithTheta(theta mat.Matrix) Kernel
	Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense)

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
	K1, K2 Kernel
}

// hyperparameters ...
func (k KernelOperator) hyperparameters() hyperparameters {
	return append(k.K1.hyperparameters(), k.K2.hyperparameters()...)
}

// Theta ...
func (k KernelOperator) Theta() mat.Matrix {
	return k.hyperparameters().Theta()
}

// Bounds ...
func (k KernelOperator) Bounds() mat.Matrix {
	return k.hyperparameters().Bounds()
}

// CloneWithTheta ...
func (k KernelOperator) CloneWithTheta(theta mat.Matrix) Kernel {
	if theta == mat.Matrix(nil) {
		return KernelOperator{K1: k.K1.CloneWithTheta(nil), K2: k.K2.CloneWithTheta(nil)}
	}
	var td = base.ToDense(theta)
	n1, _ := k.K1.Theta().Dims()
	n2, _ := k.K2.Theta().Dims()
	return KernelOperator{
		K1: k.K1.CloneWithTheta(td.Slice(0, n1, 0, 1)),
		K2: k.K2.CloneWithTheta(td.Slice(n1, n1+n2, 0, 1)),
	}
}

// Eval ...
func (k KernelOperator) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {
	panic("Eval must be implemented by wrapper")
}

// Diag ...
func (k KernelOperator) Diag(X mat.Matrix) *mat.DiagDense {
	panic("Diag must be implemented by wrapper")
}

// String ...
func (k KernelOperator) String() string {
	panic("Diag must be implemented by wrapper")
}

// IsStationary returns whether the kernel is stationary
func (k KernelOperator) IsStationary() bool {
	return k.K1.IsStationary() && k.K2.IsStationary()
}

// Sum kernel K1 + K2 of two kernels K1 and K2
type Sum struct {
	KernelOperator
}

// CloneWithTheta ...
func (k Sum) CloneWithTheta(theta mat.Matrix) Kernel {
	return &Sum{KernelOperator: k.KernelOperator.CloneWithTheta(theta).(KernelOperator)}
}

// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Sum) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {

	K1, K1g := k.K1.Eval(X, Y, evalGradient)
	K2, K2g := k.K2.Eval(X, Y, evalGradient)
	K1.Add(K1, K2)
	var Kg *t.Dense
	if evalGradient {
		s1, s2 := K1g.Shape(), K2g.Shape()
		s := t.Shape{s1[0], s1[1], s1[2] + s2[2]}
		Kg = t.NewDense(K1g.Dtype(), s)
		K1gdata, K2gdata, Kgdata := K1g.Data().([]float64), K2g.Data().([]float64), Kg.Data().([]float64)
		for i := range Kgdata {
			i2 := i % s[2]
			i1 := ((i - i2) / s[2]) % s[1]
			i0 := ((i - i2) / s[2]) / s[1]
			if i2 < s1[2] {
				Kgdata[i] = K1gdata[i0*s1[1]*s1[2]+i1*s1[2]+i2]
			} else {
				Kgdata[i] = K2gdata[i0*s2[1]*s2[2]+i1*s2[2]+(i2-s1[2])]
			}
		}
	}
	return K1, Kg
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *Sum) Diag(X mat.Matrix) *mat.DiagDense {
	K1 := k.K1.Diag(X)
	K2 := k.K2.Diag(X)
	nx, _ := K1.Dims()
	K1r, K2r := K1.RawBand(), K2.RawBand()
	for i, k1i, k2i := 0, 0, 0; i < nx; i, k1i, k2i = i+1, k1i+K1r.Stride, k2i+K2r.Stride {
		//K1.SetDiag(i, K1.At(i, i)+K2.At(i, i))
		K1r.Data[k1i] += K2r.Data[k2i]
	}
	return K1
}

// String ...
func (k *Sum) String() string {
	return k.K1.String() + " + " + k.K2.String()
}

// Product kernel K1 * K2 of two kernels K1 and K2
type Product struct {
	KernelOperator
}

// CloneWithTheta ...
func (k Product) CloneWithTheta(theta mat.Matrix) Kernel {

	return &Product{KernelOperator: k.KernelOperator.CloneWithTheta(theta).(KernelOperator)}
}

// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Product) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {

	K1, K1g := k.K1.Eval(X, Y, evalGradient)
	K2, K2g := k.K2.Eval(X, Y, evalGradient)
	K := &mat.Dense{}
	K.MulElem(K1, K2)
	var Kg *t.Dense
	if evalGradient {
		s1, s2 := K1g.Shape(), K2g.Shape()
		s := t.Shape{s1[0], s1[1], s1[2] + s2[2]}
		Kg = t.NewDense(K1g.Dtype(), s)

		K1r, K2r := K1.RawMatrix(), K2.RawMatrix()
		K1gdata, K2gdata, Kgdata := K1g.Data().([]float64), K2g.Data().([]float64), Kg.Data().([]float64)
		K1gStrides, K2gStrides, KgStrides := K1g.Strides(), K2g.Strides(), Kg.Strides()

		for i0 := 0; i0 < s[0]; i0++ {
			for i1 := 0; i1 < s[1]; i1++ {
				for i2 := 0; i2 < s[2]; i2++ {
					if i2 < s1[2] {
						Kgdata[i0*KgStrides[0]+i1*KgStrides[1]+i2*KgStrides[2]] =
							K1gdata[i0*K1gStrides[0]+i1*K1gStrides[1]+i2*K1gStrides[2]] * K2r.Data[i0*K2r.Stride+i1]
					} else {
						Kgdata[i0*KgStrides[0]+i1*KgStrides[1]+i2*KgStrides[2]] =
							K2gdata[i0*K2gStrides[0]+i1*K2gStrides[1]+(i2-s1[2])*K2gStrides[2]] * K1r.Data[i0*K1r.Stride+i1]
					}
				}
			}
		}
	}
	return K, Kg
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *Product) Diag(X mat.Matrix) *mat.DiagDense {
	K1 := k.K1.Diag(X)
	K2 := k.K2.Diag(X)
	nx, _ := K1.Dims()

	K1r, K2r := K1.RawBand(), K2.RawBand()
	for i, k1i, k2i := 0, 0, 0; i < nx; i, k1i, k2i = i+1, k1i+K1r.Stride, k2i+K2r.Stride {
		//K1.SetDiag(i, K1.At(i, i)*K2.At(i, i))
		K1r.Data[k1i] *= K2r.Data[k2i]
	}
	return K1
}

// String ...
func (k *Product) String() string {
	return k.K1.String() + " * " + k.K2.String()
}

// Exponentiation exponentiate kernel by given exponent
type Exponentiation struct {
	Kernel
	Exponent float64
}

// hyperparameters ...
func (k Exponentiation) hyperparameters() hyperparameters {
	hps := k.hyperparameters()
	params := make(hyperparameters, len(hps))
	for i, p := range hps {
		p.Name = "kernel_" + p.Name
		params[i] = p
	}
	return params
}

// Eval return the kernel k(X, Y) and optionally its gradient
func (k *Exponentiation) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {
	K, Kg := k.Kernel.Eval(X, Y, evalGradient)
	K.Apply(func(_, _ int, v float64) float64 {
		return math.Pow(v, k.Exponent)
	}, K)
	if evalGradient {
		Kdata := K.RawMatrix().Data
		Kgdata := Kg.Data().([]float64)
		for i := range Kgdata {
			Kgdata[i] *= k.Exponent * math.Pow(Kdata[i], k.Exponent-1)
		}
	}
	return K, Kg
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *Exponentiation) Diag(X mat.Matrix) *mat.DiagDense {
	K := k.Kernel.Diag(X)
	nx, _ := K.Dims()
	Kr := K.RawBand()
	for i, kri := 0, 0; i < nx; i, kri = i+1, kri+Kr.Stride {
		//K.SetDiag(i, math.Pow(K.At(i, i), k.Exponent))
		Kr.Data[kri] = math.Pow(Kr.Data[kri], k.Exponent)
	}
	return K
}

// String ...
func (k *Exponentiation) String() string {
	return fmt.Sprintf("%s ** %g", k.Kernel.String(), k.Exponent)
}

// ConstantKernel can be used as part of a product-kernel where it scales the magnitude of the other factor (kernel) or as part of a sum-kernel, where it modifies the mean of the Gaussian process.
// k(x_1, x_2) = constant_value for all x_1, x_2
type ConstantKernel struct {
	ConstantValue       float64
	ConstantValueBounds [2]float64
	StationaryKernelMixin
}

// hyperparameters ...
func (k *ConstantKernel) hyperparameters() hyperparameters {
	return hyperparameters{
		{"constant_value", &k.ConstantValue, &k.ConstantValueBounds},
	}
}

// Theta ...
func (k *ConstantKernel) Theta() mat.Matrix {
	return k.hyperparameters().Theta()
}

// Bounds ...
func (k ConstantKernel) Bounds() mat.Matrix {
	return k.hyperparameters().Bounds()
}

// CloneWithTheta ...
func (k ConstantKernel) CloneWithTheta(theta mat.Matrix) Kernel {
	clone := k
	matCopy(clone.Theta().(mat.Mutable), theta)
	return &clone
}

// Eval returns
// K : array, shape (n_samples_X, n_samples_Y)
// Kernel k(X, Y)
func (k *ConstantKernel) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {
	if X == mat.Matrix(nil) {
		panic("ConstantKernel.Eval: X is nil")
	}
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
	var Kg *t.Dense
	if evalGradient && Y == X {
		if k.hyperparameters()[0].IsFixed() {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 0})
		} else {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 1}, t.WithBacking(make([]float64, nx*nx)))
			it := Kg.Iterator()
			Kgdata := Kg.Data().([]float64)
			for i, e := it.Start(); e == nil; i, e = it.Next() {
				Kgdata[i] = k.ConstantValue
			}
		}
	}
	return K, Kg
}

// Diag returns the diagonal of the kernel k(X, X).
func (k *ConstantKernel) Diag(X mat.Matrix) (K *mat.DiagDense) {
	nx, _ := X.Dims()

	K = mat.NewDiagDense(nx, nil)
	Kr := K.RawBand()
	for i, kri := 0, 0; i < nx; i, kri = i+1, kri+Kr.Stride {
		Kr.Data[kri] = k.ConstantValue
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

// hyperparameters ...
func (k *WhiteKernel) hyperparameters() hyperparameters {
	return hyperparameters{
		{"noise_level", &k.NoiseLevel, &k.NoiseLevelBounds},
	}
}

// Theta ...
func (k *WhiteKernel) Theta() mat.Matrix {
	return k.hyperparameters().Theta()
}

// Bounds ...
func (k WhiteKernel) Bounds() mat.Matrix {
	return k.hyperparameters().Bounds()
}

// CloneWithTheta ...
func (k WhiteKernel) CloneWithTheta(theta mat.Matrix) Kernel {
	clone := k
	matCopy(k.Theta().(mat.Mutable), theta)
	return &clone
}

// Eval return the kernel k(X, Y)
func (k *WhiteKernel) Eval(X, Y mat.Matrix, evalGradient bool) (*mat.Dense, *t.Dense) {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	K := mat.NewDense(nx, ny, nil)
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)
	Kr := K.RawMatrix()
	for ix, krix := 0, 0; ix < nx; ix, krix = ix+1, krix+Kr.Stride {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			if floats.EqualApprox(Xrow, Yrow, 1e-8) {
				//K.Set(ix, iy, k.NoiseLevel)
				Kr.Data[krix+iy] = k.NoiseLevel
			}
		}
	}
	var Kg *t.Dense
	if evalGradient && Y == X {
		if k.hyperparameters()[0].IsFixed() {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 0})
		} else {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 1}, t.WithBacking(make([]float64, nx*nx)))
			it := Kg.Iterator()
			Kgdata := Kg.Data().([]float64)
			for i, e := it.Start(); e == nil; i, e = it.Next() {
				if i%(nx+1) == 0 {
					Kgdata[i] = k.NoiseLevel
				}
			}
		}
	}
	return K, Kg
}

// Diag returns the diagonal of the kernel k(X, X)
func (k *WhiteKernel) Diag(X mat.Matrix) *mat.DiagDense {
	nx, _ := X.Dims()
	K := mat.NewDiagDense(nx, nil)
	Kr := K.RawBand()
	for ix, kri := 0, 0; ix < nx; ix, kri = ix+1, kri+Kr.Stride {
		//K.SetDiag(ix, k.NoiseLevel)
		Kr.Data[kri] = k.NoiseLevel
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
	LengthScale       []float64
	LengthScaleBounds [][2]float64
	StationaryKernelMixin
	NormalizedKernelMixin
}

// hyperparameters ...
func (k *RBF) hyperparameters() hyperparameters {
	params := make(hyperparameters, len(k.LengthScale))
	for i := range k.LengthScale {
		params[i] = hyperparameter{Name: fmt.Sprintf("length_scale_%d", i), PValue: &k.LengthScale[i], PBounds: &k.LengthScaleBounds[i]}

	}
	return params
}

// Theta ...
func (k *RBF) Theta() mat.Matrix {
	return k.hyperparameters().Theta()
}

// Bounds ...
func (k RBF) Bounds() mat.Matrix {
	return k.hyperparameters().Bounds()
}

// CloneWithTheta ...
func (k RBF) CloneWithTheta(theta mat.Matrix) Kernel {
	clone := k
	clone.LengthScale = make([]float64, len(k.LengthScale))
	copy(clone.LengthScale, k.LengthScale)
	clone.LengthScaleBounds = make([][2]float64, len(k.LengthScaleBounds))
	copy(clone.LengthScaleBounds, k.LengthScaleBounds)
	matCopy(clone.Theta().(mat.Mutable), theta)
	return &clone
}

// Eval return the kernel k(X, Y)
func (k *RBF) Eval(X, Y mat.Matrix, evalGradient bool) (K *mat.Dense, Kg *t.Dense) {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	var scale func([]float64, int) float64
	switch len(k.LengthScale) {
	case 1:
		scale = func(X []float64, feat int) float64 { return X[feat] / k.LengthScale[0] }
	default:
		if len(k.LengthScale) != nfeat {
			panic("LengthScale has wrong dimension")
		}
		scale = func(X []float64, feat int) float64 { return X[feat] / k.LengthScale[feat] }
	}

	K = mat.NewDense(nx, ny, nil)
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)

	// K=np.exp(-0.5 * cdist(X/K.length_scale,Y/K.length_scale,'sqeuclidean'))
	Kr := K.RawMatrix()
	for ix, krix := 0, 0; ix < nx; ix, krix = ix+1, krix+Kr.Stride {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			var d2 float64
			for feat := 0; feat < nfeat; feat++ {
				d := scale(Xrow, feat) - scale(Yrow, feat)
				d2 += d * d
			}
			//K.Set(ix, iy, math.Exp(-.5*d2))
			Kr.Data[krix+iy] = math.Exp(-.5 * d2)

		}
	}
	if evalGradient && Y == X {
		if k.hyperparameters()[0].IsFixed() {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 0}, t.WithBacking(nil))
		} else {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 1}, t.WithBacking(make([]float64, nx*nx)))
			it := Kg.Iterator()
			Kgdata := Kg.Data().([]float64)
			Kdata := K.RawMatrix().Data
			xr, xc, xd := mat.NewVecDense(nfeat, nil), mat.NewVecDense(nfeat, nil), mat.NewVecDense(nfeat, nil)
			for i, e := it.Start(); e == nil; i, e = it.Next() {
				r, c := i/nx, i%nx
				mat.Row(xr.RawVector().Data, r, X)
				mat.Row(xc.RawVector().Data, c, X)
				xd.SubVec(xr, xc)
				switch len(k.LengthScale) {
				case 1:
					xd.ScaleVec(1/k.LengthScale[0], xd)
				case nfeat:
					xd.DivElemVec(xd, mat.NewVecDense(nfeat, k.LengthScale))
				default:
					panic("dim error")
				}
				Kgdata[i] = Kdata[i] * mat.Dot(xd, xd)
			}
		}
	}
	return
}

// IsAnisotropic ...
func (k *RBF) IsAnisotropic() bool {
	return len(k.LengthScale) > 1
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

// hyperparameters ...
func (k *DotProduct) hyperparameters() hyperparameters {
	return hyperparameters{
		{"sigma_0", &k.Sigma0, &k.Sigma0Bounds},
	}
}

// Theta ...
func (k *DotProduct) Theta() mat.Matrix {
	return k.hyperparameters().Theta()
}

// Bounds ...
func (k DotProduct) Bounds() mat.Matrix {
	return k.hyperparameters().Bounds()
}

// CloneWithTheta ...
func (k DotProduct) CloneWithTheta(theta mat.Matrix) Kernel {
	clone := k
	matCopy(clone.Theta().(mat.Mutable), theta)
	return &clone
}

// Eval return the kernel k(X, Y)
func (k *DotProduct) Eval(X, Y mat.Matrix, evalGradient bool) (K *mat.Dense, Kg *t.Dense) {
	nx, nfeat := X.Dims()
	if Y == mat.Matrix(nil) {
		Y = X
	}
	ny, _ := Y.Dims()
	K = mat.NewDense(nx, ny, nil)
	s2 := k.Sigma0 * k.Sigma0
	Xrow := make([]float64, nfeat)
	Yrow := make([]float64, nfeat)
	Kr := K.RawMatrix()
	for ix, krix := 0, 0; ix < nx; ix, krix = ix+1, krix+Kr.Stride {
		mat.Row(Xrow, ix, X)
		for iy := 0; iy < ny; iy++ {
			mat.Row(Yrow, iy, Y)
			//K.Set(ix, iy, s2+mat.Dot(mat.NewVecDense(nfeat, Xrow), mat.NewVecDense(nfeat, Yrow)))
			Kr.Data[krix+iy] = s2 + mat.Dot(mat.NewVecDense(nfeat, Xrow), mat.NewVecDense(nfeat, Yrow))
		}
	}
	if evalGradient && Y == X {
		if k.hyperparameters()[0].IsFixed() {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 0})
		} else {
			Kg = t.NewDense(t.Float64, t.Shape{nx, nx, 1}, t.WithBacking(make([]float64, nx*nx)))
			it := Kg.Iterator()
			Kgdata := Kg.Data().([]float64)
			for i, e := it.Start(); e == nil; i, e = it.Next() {
				Kgdata[i] = 2 * k.Sigma0 * k.Sigma0
			}
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
	Kr := K.RawBand()
	for i, kri := 0, 0; i < n; i, kri = i+1, kri+Kr.Stride {
		mat.Row(Xrow, i, X)
		Vrow := mat.NewVecDense(nfeat, Xrow)
		//K.SetDiag(i, s2+mat.Dot(Vrow, Vrow))
		Kr.Data[kri] = s2 + mat.Dot(Vrow, Vrow)
	}
	return
}

// IsStationary returns whether the kernel is stationary
func (k *DotProduct) IsStationary() bool { return false }

// String returns  string representation of kernel
func (k *DotProduct) String() string {
	return fmt.Sprintf("DotProduct(sigma_0=%.3g)", k.Sigma0)
}
