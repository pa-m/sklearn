package neuralnetwork

import "gonum.org/v1/gonum/mat"

// Matrix interface (identical to gonum/mat one's)
type Matrix = mat.Matrix

/*type Matrix interface {
	Dims() (r, c int)
	At(i, j int) float64
	T() Matrix
}*/
