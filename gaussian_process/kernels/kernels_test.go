package kernels

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/pa-m/randomkit"
)

var _ = []Kernel{&ConstantKernel{}, &WhiteKernel{}, &RBF{}, &DotProduct{}, &Sum{}, &Product{}, &Exponentiation{}}

type Float64er interface{ Float64() float64 }

func sample(state Float64er, r, c int) *mat.Dense {
	ret := mat.NewDense(r, c, nil)
	data := ret.RawMatrix().Data
	for i := range data {
		data[i] = state.Float64()
	}
	return ret
}

func ExampleConstantKernel() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	K := &ConstantKernel{ConstantValue: 1.23}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())
	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=1.11**2, stationary:true
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	// K(X,Y)=
	// ⎡1.23000000  1.23000000  1.23000000⎤
	// ⎢1.23000000  1.23000000  1.23000000⎥
	// ⎣1.23000000  1.23000000  1.23000000⎦
	// K(X,X)=
	// ⎡1.23000000  0.00000000  0.00000000⎤
	// ⎢0.00000000  1.23000000  0.00000000⎥
	// ⎣0.00000000  0.00000000  1.23000000⎦

}
func ExampleWhiteKernel() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	K := &WhiteKernel{NoiseLevel: 1.23}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=WhiteKernel(noise_level=1.23), stationary:true
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	// K(X,Y)=
	// ⎡0.00000000  0.00000000  0.00000000⎤
	// ⎢0.00000000  0.00000000  0.00000000⎥
	// ⎣0.00000000  0.00000000  0.00000000⎦
	// K(X,X)=
	// ⎡1.23000000  0.00000000  0.00000000⎤
	// ⎢0.00000000  1.23000000  0.00000000⎥
	// ⎣0.00000000  0.00000000  1.23000000⎦
}

func ExampleRBF() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	// K=DotProduct(sigma_0=1.23)
	K := &RBF{LengthScale: []float64{1.23}}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=RBF([1.23]), stationary:true
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	// K(X,Y)=
	// ⎡0.93799022  0.98903690  0.99959124⎤
	// ⎢0.98800335  0.93194636  0.89898014⎥
	// ⎣0.97852658  0.91710014  0.86874975⎦
	// K(X,X)=
	// ⎡1.00000000  0.00000000  0.00000000⎤
	// ⎢0.00000000  1.00000000  0.00000000⎥
	// ⎣0.00000000  0.00000000  1.00000000⎦

}

func ExampleDotProduct() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	// K=DotProduct(sigma_0=1.23)
	K := &DotProduct{Sigma0: 1.23}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=DotProduct(sigma_0=1.23), stationary:false
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	// K(X,Y)=
	// ⎡1.83949046  2.06648366  2.18129373⎤
	// ⎢1.61739557  1.67584723  1.72011212⎥
	// ⎣1.57214338  1.62088154  1.63769147⎦
	// K(X,X)=
	// ⎡2.20567473  0.00000000  0.00000000⎤
	// ⎢0.00000000  1.60430500  0.00000000⎥
	// ⎣0.00000000  0.00000000  1.54296371⎦

}

func ExampleSum() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	// K=DotProduct(sigma_0=1.23)
	K := &Sum{KernelOperator{k1: &ConstantKernel{ConstantValue: 1.23}, k2: &WhiteKernel{NoiseLevel: 1.23}}}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=1.11**2 + WhiteKernel(noise_level=1.23), stationary:true
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	// K(X,Y)=
	// ⎡1.23000000  1.23000000  1.23000000⎤
	// ⎢1.23000000  1.23000000  1.23000000⎥
	// ⎣1.23000000  1.23000000  1.23000000⎦
	// K(X,X)=
	// ⎡2.46000000  0.00000000  0.00000000⎤
	// ⎢0.00000000  2.46000000  0.00000000⎥
	// ⎣0.00000000  0.00000000  2.46000000⎦

}

func ExampleProduct() {
	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	// K=DotProduct(sigma_0=1.23)
	K := &Product{KernelOperator{
		k1: &ConstantKernel{ConstantValue: 1.23},
		k2: &DotProduct{Sigma0: 1.23}},
	}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=1.11**2 * DotProduct(sigma_0=1.23), stationary:false
	// X=
	// ⎡0.41702200  0.72032449⎤
	// ⎢0.00011437  0.30233257⎥
	// ⎣0.14675589  0.09233859⎦
	// Y=
	// ⎡0.18626021  0.34556073⎤
	// ⎢0.39676747  0.53881673⎥
	// ⎣0.41919451  0.68521950⎦
	//K(X,Y)=
	// ⎡2.26257327  2.54177490  2.68299128⎤
	// ⎢1.98939655  2.06129209  2.11573791⎥
	// ⎣1.93373635  1.99368430  2.01436051⎦
	// K(X,X)=
	// ⎡2.71297992  0.00000000  0.00000000⎤
	// ⎢0.00000000  1.97329515  0.00000000⎥
	// ⎣0.00000000  0.00000000  1.89784536⎦

}

func ExampleKernel_Theta() {
	kernel := &Product{KernelOperator{
		k1: &ConstantKernel{ConstantValue: 1., ConstantValueBounds: [2]float64{1e-3, 1e3}},
		k2: &RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}},
	}}
	fmt.Printf("%.8f\n", mat.Formatted(kernel.Theta()))
	// Output:
	// ⎡0.00000000⎤
	// ⎣2.30258509⎦
}
func ExampleKernel_Bounds() {
	kernel := &Product{KernelOperator{
		k1: &ConstantKernel{ConstantValue: 1., ConstantValueBounds: [2]float64{1e-3, 1e3}},
		k2: &RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}},
	}}
	fmt.Printf("%.8f\n", mat.Formatted(kernel.Bounds()))
	// Output:
	// ⎡-6.90775528   6.90775528⎤
	// ⎣-4.60517019   4.60517019⎦
}

func assertEq(t testing.TB, expected, actual mat.Matrix, msg string) {
	diff := mat.DenseCopyOf(expected)
	diff.Sub(diff, actual)
	if mat.Norm(diff, 1) > 1e-6 {
		t.Errorf("%s\nexpected:\n%g\ngot:\n%g\n", msg, mat.Formatted(expected), mat.Formatted(actual))
	}
}

func TestWhiteKernel(t *testing.T) {
	kernel := &WhiteKernel{NoiseLevel: 1., NoiseLevelBounds: [2]float64{1e-5, 1e5}}
	assertEq(
		t,
		mat.NewDense(1, 1, []float64{0}),
		kernel.Theta(),
		"wrong theta")
	assertEq(
		t,
		mat.NewDense(1, 2, []float64{math.Log(1e-5), math.Log(1e5)}),
		kernel.Bounds(),
		"wrong bounds")
}

func TestExponentiation(t *testing.T) {
	var kernel Kernel
	kernel = &Product{KernelOperator{
		k1: &ConstantKernel{ConstantValue: 1., ConstantValueBounds: [2]float64{1e-3, 1e3}},
		k2: &RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}},
	}}
	kernel = &Exponentiation{kernel, 2.}

	assertEq(
		t,
		mat.NewDense(2, 1, []float64{0, math.Log(10)}),
		kernel.Theta(),
		"wrong theta")
	assertEq(
		t,
		mat.NewDense(2, 2, []float64{math.Log(1e-3), math.Log(1e3), math.Log(1e-2), math.Log(1e2)}),
		kernel.Bounds(),
		"wrong bounds")

	// np.random.seed(1)
	state := randomkit.NewRandomkitSource(1)
	// X=np.reshape(np.random.sample(6),(3,2))
	X, Y := sample(state, 3, 2), sample(state, 3, 2)
	actual := kernel.Eval(X, Y)
	assertEq(
		t,
		mat.NewDense(3, 3, []float64{
			0.99806489, 0.9996665, 0.99998763,
			0.99963488, 0.99786969, 0.99678288,
			0.9993434, 0.99738494, 0.99575174,
		}),
		actual,
		"wrong K(X,Y)")
	assertEq(
		t,
		mat.NewDiagDense(3, []float64{1, 1, 1}),
		kernel.Diag(X),
		"wrong K(X,X)")
}

func TestKernel_CloneWithTheta(t *testing.T) {
	k1 := &ConstantKernel{ConstantValue: 1., ConstantValueBounds: [2]float64{1e-3, 1e3}}
	k2 := &RBF{LengthScale: []float64{10}, LengthScaleBounds: [][2]float64{{1e-2, 1e2}}}
	var kernel Kernel
	kernel = &Product{KernelOperator{
		k1: k1,
		k2: k2,
	}}
	newTheta := mat.NewDense(2, 1, []float64{1.1, 1.2})

	clone := kernel.CloneWithTheta(newTheta)
	assertEq(
		t,
		newTheta,
		clone.Theta(),
		"wrong theta")
}
