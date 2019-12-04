package neuralnetwork

import (
	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

// MLPRegressor ...
type MLPRegressor struct{ BaseMultilayerPerceptron64 }

// Regressors is the list of regressors in this package
var Regressors = []base.Predicter{&MLPRegressor{}}

// NewMLPRegressor returns a *MLPRegressor with defaults
// activation is one of identity,logistic,tanh,relu
// solver is on of sgd,adam  defaults to "adam"
// Alpha is the regularization parameter
func NewMLPRegressor(hiddenLayerSizes []int, activation string, solver string, Alpha float64) *MLPRegressor {
	mlp := &MLPRegressor{
		BaseMultilayerPerceptron64: *NewBaseMultilayerPerceptron64(),
	}
	mlp.HiddenLayerSizes = hiddenLayerSizes
	mlp.Activation = activation
	mlp.Solver = solver
	mlp.Alpha = Alpha
	return mlp
}

// IsClassifier returns false for MLPRegressor
func (*MLPRegressor) IsClassifier() bool { return false }

// PredicterClone allow clone predicter for pipeline on model_selection
func (mlp *MLPRegressor) PredicterClone() base.Predicter {
	if mlp == nil {
		return nil
	}
	clone := *mlp
	if sourceCloner, ok := clone.RandomState.(base.SourceCloner); ok && sourceCloner != base.SourceCloner(nil) {
		clone.RandomState = sourceCloner.SourceClone()
	}
	return &clone
}

// Fit ...
func (mlp *MLPRegressor) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	mlp.fit(X.RawMatrix(), Y.RawMatrix(), false)
	return mlp
}

// Predict return the forward result
func (mlp *MLPRegressor) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, mlp.GetNOutputs(), nil)
	}

	mlp.BaseMultilayerPerceptron64.predict(base.ToDense(X).RawMatrix(), Y.RawMatrix())
	return base.FromDense(Ymutable, Y)
}

// Score for MLPRegressor returns R2Score
func (mlp *MLPRegressor) Score(X, Y mat.Matrix) float64 {
	nSamples, _ := X.Dims()
	nOutputs := mlp.NOutputs
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	mlp.Predict(X, Ypred)
	return r2Score64(base.ToDense(Y).RawMatrix(), Ypred.RawMatrix())
}

// MLPClassifier ...
type MLPClassifier struct{ BaseMultilayerPerceptron64 }

// NewMLPClassifier returns a *MLPClassifier with defaults
// activation is one of logistic,tanh,relu
// solver is on of agd,adagrad,rmsprop,adadelta,adam (one of the keys of base.Solvers) defaults to "adam"
// Alpha is the regularization parameter
// lossName is one of square,log,cross-entropy (one of the keys of lm.LossFunctions) defaults to "log"
func NewMLPClassifier(hiddenLayerSizes []int, activation string, solver string, Alpha float64) *MLPClassifier {
	mlp := &MLPClassifier{
		BaseMultilayerPerceptron64: *NewBaseMultilayerPerceptron64(),
	}
	mlp.HiddenLayerSizes = hiddenLayerSizes
	mlp.Activation = activation
	mlp.Solver = solver
	mlp.Alpha = Alpha
	return mlp
}

// PredicterClone returns an (possibly unfitted) copy of predicter
func (mlp *MLPClassifier) PredicterClone() base.Predicter {
	clone := *mlp
	return &clone
}

// IsClassifier returns true for MLPClassifier
func (*MLPClassifier) IsClassifier() bool { return true }

// Fit ...
func (mlp *MLPClassifier) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	mlp.BaseMultilayerPerceptron64.Fit(X, Y)
	return mlp
}

// Predict return the forward result for MLPClassifier
func (mlp *MLPClassifier) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Y := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Y.IsEmpty() {
		*Y = *mat.NewDense(nSamples, mlp.GetNOutputs(), nil)
	}

	if mlp.LossFuncName == "" {
		mlp.LossFuncName = "binary_log_loss"
	}
	yr, _ := Y.Dims()
	if yr == 0 {
		panic("Y must be preallocated")
	}
	mlp.BaseMultilayerPerceptron64.Predict(X, Y)
	return base.FromDense(Ymutable, Y)
}

// Score for MLPClassifier computes accuracy score
func (mlp *MLPClassifier) Score(Xmatrix, Ymatrix mat.Matrix) float64 {
	X, Y := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	Ypred := mat.NewDense(X.RawMatrix().Rows, Y.RawMatrix().Cols, nil)
	mlp.Predict(X, Ypred)

	return accuracyScore64(Y.RawMatrix(), Ypred.RawMatrix())
}
