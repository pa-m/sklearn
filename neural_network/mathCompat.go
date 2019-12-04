package neuralnetwork

import (
	m64 "math"

	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"

	m32 "github.com/chewxy/math32"
)

type floatXX = float32

// M32 has funcs for float32 math
var M32 = struct {
	Ceil       func(float32) float32
	Sqrt       func(float32) float32
	Pow        func(float32, float32) float32
	IsInf      func(float32, int) bool
	Abs        func(float32) float32
	Exp        func(float32) float32
	Tanh       func(float32) float32
	Log        func(float32) float32
	Log1p      func(float32) float32
	MaxFloat32 float32
	Inf        func(int) float32
	IsNaN      func(float32) bool
	Nextafter  func(x, y float32) float32
	MaxFloatXX floatXX
}{
	Ceil: m32.Ceil, Sqrt: m32.Sqrt, Pow: m32.Pow, IsInf: m32.IsInf, Abs: m32.Abs, Exp: m32.Exp, Tanh: m32.Tanh, Log: m32.Log, Log1p: m32.Log1p,
	MaxFloat32: m32.MaxFloat32, Inf: m32.Inf, IsNaN: m32.IsNaN, Nextafter: m32.Nextafter, MaxFloatXX: m32.MaxFloat32}

// M64 has funcs for float64 math
var M64 = struct {
	Ceil       func(float64) float64
	Sqrt       func(float64) float64
	Pow        func(float64, float64) float64
	IsInf      func(float64, int) bool
	Abs        func(float64) float64
	Exp        func(float64) float64
	Tanh       func(float64) float64
	Log        func(float64) float64
	Log1p      func(float64) float64
	MaxFloat64 float64
	Inf        func(int) float64
	IsNaN      func(float64) bool
	Nextafter  func(x, y float64) float64
}{Ceil: m64.Ceil, Sqrt: m64.Sqrt, Pow: m64.Pow, IsInf: m64.IsInf, Abs: m64.Abs, Exp: m64.Exp, Tanh: m64.Tanh, Log: m64.Log, Log1p: m64.Log1p,
	MaxFloat64: m64.MaxFloat64, Inf: m64.Inf, IsNaN: m64.IsNaN, Nextafter: m64.Nextafter}

// MXX has funcs for floatXX math
var MXX = M32

type blas32Vector = blas32.Vector
type blas64Vector = blas64.Vector
type blasXXVector = blas32.Vector

func dot32(n, xinc, yinc int, x, y []float32) float32 {
	return blas32.Dot(blas32.Vector{N: n, Inc: xinc, Data: x}, blas32.Vector{Inc: yinc, Data: y})
}
func dot64(n, xinc, yinc int, x, y []float64) float64 {
	return blas64.Dot(blas64.Vector{N: n, Inc: xinc, Data: x}, blas64.Vector{N: n, Inc: yinc, Data: y})
}

var gemm32 = blas32.Gemm
var gemm64 = blas64.Gemm

// axpy32 adds x scaled by alpha to y:
//  y[i] += alpha*x[i] for all i.
func axpy32(n int, alpha float32, X, Y []float32) {
	blas32.Axpy(alpha, blas32.Vector{N: n, Inc: 1, Data: X}, blas32.Vector{N: n, Inc: 1, Data: Y})
}

// axpy64 adds x scaled by alpha to y:
//  y[i] += alpha*x[i] for all i.
func axpy64(n int, alpha float64, X, Y []float64) {
	blas64.Axpy(alpha, blas64.Vector{N: n, Data: X, Inc: 1}, blas64.Vector{N: n, Data: Y, Inc: 1})
}

// MaxIdx32 ...
func MaxIdx32(a []float32) int {
	var mi int
	for i := range a {
		if a[i] > a[mi] {
			mi = i
		}
	}
	return mi
}

// MaxIdx64 ...
func MaxIdx64(a []float64) int {
	var mi int
	for i := range a {
		if a[i] > a[mi] {
			mi = i
		}
	}
	return mi
}

// MaxIdxXX ...
var MaxIdxXX = MaxIdx32
var toLogitsXX = toLogits32
