package modelselection

import (
	"fmt"
	"sort"

	"github.com/pa-m/sklearn/datasets"
	linearModel "github.com/pa-m/sklearn/linear_model"
	"github.com/pa-m/sklearn/metrics"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func ExampleCrossValidate() {
	// example adapted from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
	randomState := rand.New(rand.NewSource(5))
	diabetes := datasets.LoadDiabetes()
	X, y := diabetes.X.Slice(0, 150, 0, diabetes.X.RawMatrix().Cols).(*mat.Dense), diabetes.Y.Slice(0, 150, 0, 1).(*mat.Dense)
	lasso := linearModel.NewLasso()
	scorer := func(Y, Ypred *mat.Dense) float64 {
		e := metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
		return e
	}
	cvresults := CrossValidate(lasso, X, y, nil, scorer, &KFold{NSplits: 3, Shuffle: true, RandomState: randomState}, 3)
	sort.Sort(cvresults)
	fmt.Printf("%.8f", cvresults.TestScore)
	// Output:
	// [0.38916221 0.33615684 0.28314305]

}
