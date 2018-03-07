package neuralNetwork

import (
	"../base"
)

// matTranspose is a matrix transposed from its initializer
type (
	matTranspose = base.MatTranspose
	matApply     = base.MatApply1

	// onesAddedMat is a matrix with an initial column of ones has been added
	onesAddedMat = base.MatOnesPrepended

	// MatFirstColumnRemoved is a matrix whose an initial column has been removed respective to its initializer
	MatFirstColumnRemoved = base.MatFirstColumnRemoved

	matApply0 = base.MatApply0

	matApply2 = base.MatApply2

	matSub = base.MatSub

	matMulElem = base.MatMulElem

	matScale = base.MatScaled

	matOneMinus = base.MatOneMinus
)
