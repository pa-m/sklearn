package linearmodel

import (
	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// LogisticRegression WIP
type LogisticRegression struct {
	RegularizedRegression
	LabelBinarizer *preprocessing.LabelBinarizer
	nOutputs       int
}

// NewLogisticRegression create and init a *LogisticRegression
func NewLogisticRegression() *LogisticRegression {
	regr := &LogisticRegression{}
	//regr.StepSize = 1e-4
	regr.Tol = 1e-6
	//regr.Optimizer = base.NewAdamOptimizer()
	regr.FitIntercept = true
	regr.Normalize = false
	regr.ActivationFunction = base.Logistic{}
	regr.LossFunction = CrossEntropyLoss
	regr.Options.GOMethodCreator = func() optimize.Method { return &optimize.LBFGS{} }
	return regr
}

// PredicterClone for LogisticRegression
func (regr *LogisticRegression) PredicterClone() base.Predicter {
	clone := *regr
	return &clone
}

// IsClassifier returns true for LogisticRegression
func (regr *LogisticRegression) IsClassifier() bool { return true }

// EncodeLabels applies a onehotencoder if Ytrue has only one column
func (regr *LogisticRegression) EncodeLabels(Ytrue *mat.Dense) *mat.Dense {
	Y := Ytrue
	_, nOutputs := Ytrue.Dims()

	if nOutputs == 1 {
		hasMoreThanTwoValues := func() bool {
			cmap := make(map[float64]bool)
			y := Ytrue.RawMatrix()
			for i, yp := 0, 0; i < y.Rows; i, yp = i+1, yp+y.Stride {
				v := y.Data[yp]
				if _, ok := cmap[v]; !ok {
					cmap[v] = true
					if len(cmap) > 2 {
						return true
					}
				}
			}
			return false
		}
		if hasMoreThanTwoValues() {
			regr.LabelBinarizer = &preprocessing.LabelBinarizer{}
			_, Y = regr.LabelBinarizer.FitTransform(nil, Ytrue)
		}
	}
	return Y
}

// Fit for LogisticRegression
func (regr *LogisticRegression) Fit(Xmatrix, Ymatrix mat.Matrix) base.Fiter {
	X0, Y0 := base.ToDense(Xmatrix), base.ToDense(Ymatrix)
	regr.nOutputs = Y0.RawMatrix().Cols
	Y := regr.EncodeLabels(Y0)
	return regr.RegularizedRegression.Fit(X0, Y)
}

// GetNOutputs returns output columns number for Y to pass to predict
func (regr *LogisticRegression) GetNOutputs() int {
	return regr.nOutputs
}

// PredictProba predicts probability of y=1 for X using Coef
func (regr *LogisticRegression) PredictProba(X mat.Matrix, Y *mat.Dense) {
	regr.DecisionFunction(X, Y)
	Y.Apply(func(i int, o int, y float64) float64 {
		return regr.ActivationFunction.F(y)
	}, Y)
}

// Predict predicts y for X using Coef
func (regr *LogisticRegression) Predict(X mat.Matrix, Ymutable mat.Mutable) *mat.Dense {
	Ycls := base.ToDense(Ymutable)
	nSamples, _ := X.Dims()
	if Ycls.IsZero() {
		*Ycls = *mat.NewDense(nSamples, regr.GetNOutputs(), nil)
	}

	var Y = Ycls
	if regr.LabelBinarizer != nil {
		Y = &mat.Dense{}
	}
	regr.PredictProba(X, Y)
	//nSamples, nOutputs := Y.Dims()
	if regr.LabelBinarizer != nil {
		_, Ycls1 := regr.LabelBinarizer.InverseTransform(nil, Y)
		if Ycls.IsZero() {
			Ycls.SetRawMatrix(Ycls1.RawMatrix())
		} else {
			Ycls.Copy(Ycls1)
		}
	} else {
		Ycls.Apply(func(_, _ int, y float64) float64 {
			if y > .5 {
				return 1
			}
			return 0
		}, Ycls)
	}
	return base.FromDense(Ymutable, Ycls)
}
