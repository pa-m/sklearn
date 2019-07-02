package kernels

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"github.com/pa-m/randomkit"
)

var _ = []Kernel{&ConstantKernel{}, &WhiteKernel{}, &RBF{}, &DotProduct{}}

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
	// K=1.23**2, stationary:true
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
	K := &RBF{LengthScale: 1.23}
	fmt.Printf("K=%s, stationary:%v\n", K, K.IsStationary())

	fmt.Printf("X=\n%.8f\nY=\n%.8f\nK(X,Y)=\n%.8f\nK(X,X)=\n%.8f\n", mat.Formatted(X), mat.Formatted(Y), mat.Formatted(K.Eval(X, Y)), mat.Formatted(K.Diag(X)))
	// Output:
	// K=RBF(1.23), stationary:true
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
