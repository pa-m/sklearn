package kernels

import (
	"fmt"
	"github.com/pa-m/sklearn/base"
	"math"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

// Hyperparameter specification
// for no value_type is mat.Vector
// bounds is [2]float64
// IsFixed() is true if bounds are equal
type Hyperparameter struct {
	Name string
	PValue *float64
	PBounds *[2]float64
}
// IsFixed return true when bounds are equal or missing
func (param Hyperparameter)IsFixed()bool {
	return param.PBounds==nil || (*param.PBounds)[1]==(*param.PBounds)[0]
}

// Kernel interface
type Kernel interface {
	Hyperparameters()[]Hyperparameter
	Theta()mat.Matrix
	Bounds()mat.Matrix
	CloneWithTheta(theta mat.Matrix) Kernel
	Eval(X, Y mat.Matrix) *mat.Dense
	Diag(X mat.Matrix) (K *mat.DiagDense)
	IsStationary() bool
	String() string

	//GetParams() map[string]interface{}
	//SetParams(interface{})
}

func kernelTheta(k Kernel) (t mat.Matrix) {
	params:=k.Hyperparameters()
	notFixed:=[]Hyperparameter{}
	for _,p:=range params{
		if !p.IsFixed(){notFixed=append(notFixed,p)}
	}
	return matFromFunc{
		r:len(notFixed),
		c:1,
		at:func(i,j int)float64{
			return math.Log(*notFixed[i].PValue)
		},
		set:func(i,j int,v float64){
			*notFixed[i].PValue=math.Exp(v)
		},
	}
}
func kernelBounds(k Kernel) (t mat.Matrix) {
	params:=k.Hyperparameters()
	notFixed:=[]Hyperparameter{}
	for _,p:=range params{
		if !p.IsFixed(){notFixed=append(notFixed,p)}
	}
	return matFromFunc{
		r:len(notFixed),
		c:2,
		at:func(i,j int)float64{
			return math.Log((*notFixed[i].PBounds)[j])
		},
		set:func(i,j int,v float64){
			(*notFixed[i].PBounds)[j]=math.Exp(v)
		},
	}
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

// Hyperparameters ...
func (k KernelOperator)Hyperparameters()[]Hyperparameter {
	return append(k.k1.Hyperparameters(),k.k2.Hyperparameters()...)
}
// Theta ...
func (k KernelOperator)Theta()mat.Matrix {
	return matVStack([]mat.Matrix{k.k1.Theta(),k.k2.Theta()})
}
// Bounds ...
func (k KernelOperator)Bounds()mat.Matrix {
	return matVStack([]mat.Matrix{k.k1.Bounds(),k.k2.Bounds()})
}
// CloneWithTheta ...
func (k KernelOperator) CloneWithTheta(theta mat.Matrix) Kernel {
	var td=base.ToDense(theta)
	n1,_:=k.k1.Theta().Dims()
	n2,_:=k.k2.Theta().Dims()
	return &KernelOperator{
		k1:k.k1.CloneWithTheta(td.Slice(0,n1,0,1)),
		k2:k.k2.CloneWithTheta(td.Slice(n1,n1+n2,0,1)),
	}
}
// Eval ...
func (k KernelOperator) Eval(X,Y mat.Matrix)*mat.Dense{
	panic("Eval must be implemented by wrapper")
}
// Diag ...
func (k KernelOperator) Diag(X mat.Matrix)*mat.DiagDense{
	panic("Diag must be implemented by wrapper")
}
// String ...
func (k KernelOperator) String()string{
	panic("Diag must be implemented by wrapper")
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
// Hyperparameters ...
func (k Exponentiation)Hyperparameters()[]Hyperparameter {
	hps:=k.Hyperparameters()
	params:=make([]Hyperparameter,len(hps))
	for i,p:=range hps{
		p.Name="kernel_" +p.Name
		params[i]=p
	}
	return params
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
// Hyperparameters ...
func (k *ConstantKernel)Hyperparameters()[]Hyperparameter {
	return []Hyperparameter{
		{"constant_value", &k.ConstantValue,&k.ConstantValueBounds},
	}
}
// Theta ...
func (k*ConstantKernel)Theta()mat.Matrix{
	return kernelTheta(k)
}
// Bounds ...
func (k*ConstantKernel)Bounds()mat.Matrix{
	return kernelBounds(k)
}
// CloneWithTheta ...
func (k*ConstantKernel)CloneWithTheta(theta mat.Matrix)Kernel{
	clone:=*k;
	matCopy(k.Theta().(mat.Mutable),theta)
	return &clone
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
// Hyperparameters ...
func (k *WhiteKernel)Hyperparameters()[]Hyperparameter {
	return []Hyperparameter{
		{"noise_level", &k.NoiseLevel,&k.NoiseLevelBounds},
	}
}
// Theta ...
func (k*WhiteKernel)Theta()mat.Matrix{
	return kernelTheta(k)
}
// Bounds ...
func (k*WhiteKernel)Bounds()mat.Matrix{
	return kernelBounds(k)
}
// CloneWithTheta ...
func (k*WhiteKernel)CloneWithTheta(theta mat.Matrix)Kernel{
	clone:=*k;
	matCopy(k.Theta().(mat.Mutable),theta)
	return &clone
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
	LengthScale       []float64
	LengthScaleBounds [][2]float64
	StationaryKernelMixin
	NormalizedKernelMixin
}
// Hyperparameters ...
func (k *RBF)Hyperparameters()[]Hyperparameter {
	params:=make([]Hyperparameter,len(k.LengthScale))
	for i:=range k.LengthScale{
		params=append(params,Hyperparameter{Name:fmt.Sprintf("length_scale_%d",i),PValue:&k.LengthScale[i],PBounds:&k.LengthScaleBounds[i]})

	}
	return params
}
// Theta ...
func (k*RBF)Theta()mat.Matrix{
	return kernelTheta(k)
}
// Bounds ...
func (k*RBF)Bounds()mat.Matrix{
	return kernelBounds(k)
}
// CloneWithTheta ...
func (k*RBF)CloneWithTheta(theta mat.Matrix)Kernel{
	clone:=*k;
	clone.LengthScale=make([]float64,len(k.LengthScale))
	copy(clone.LengthScale,k.LengthScale)
	clone.LengthScaleBounds=make([][2]float64,len(k.LengthScaleBounds))
	copy(clone.LengthScaleBounds,k.LengthScaleBounds)
	matCopy(k.Theta().(mat.Mutable),theta)
	return &clone
}

// Eval return the kernel k(X, Y)
func (k *RBF) Eval(X, Y mat.Matrix) (K *mat.Dense) {
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
	return len(k.LengthScale)>1
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
// Hyperparameters ...
func (k *DotProduct)Hyperparameters()[]Hyperparameter {
	return []Hyperparameter{
		{"sigma_0",&k.Sigma0,&k.Sigma0Bounds},
	}
}
// Theta ...
func (k*DotProduct)Theta()mat.Matrix{
	return kernelTheta(k)
}
// Bounds ...
func (k*DotProduct)Bounds()mat.Matrix{
	return kernelBounds(k)
}
// CloneWithTheta ...
func (k*DotProduct)CloneWithTheta(theta mat.Matrix)Kernel{
	clone:=*k;
	matCopy(k.Theta().(mat.Mutable),theta)
	return &clone
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

