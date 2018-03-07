package neuralNetwork

import (
	"../base"
	"gonum.org/v1/gonum/mat"
)

// matTranspose is a matrix transposed from its initializer
type matTranspose = base.MatTranspose

type matApply = base.MatApply1

// onesAddedMat is a matrix with an initial column of ones has been added
type onesAddedMat = base.MatOnesPrepended

// MatFirstColumnRemoved is a matrix whose an initial column has been removed respective to its initializer
type MatFirstColumnRemoved = base.MatFirstColumnRemoved

type matApply0 = base.MatApply0

type matApply2 = base.MatApply2

type matSub = base.MatSub

type matMulElem = base.MatMulElem

type matScale struct {
	mat.Matrix
	float64
}

func (m matScale) Dims() (int, int)    { return m.Matrix.Dims() }
func (m matScale) At(i, j int) float64 { return m.Matrix.At(i, j) * m.float64 }
func (m matScale) T() mat.Matrix       { return matTranspose{Matrix: m} }

type matOneMinus struct {
	mat.Matrix
}

func (m matOneMinus) Dims() (int, int)    { return m.Dims() }
func (m matOneMinus) At(i, j int) float64 { return 1. - m.At(i, j) }
func (m matOneMinus) T() mat.Matrix       { return matTranspose{Matrix: m} }
