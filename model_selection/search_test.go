package modelselection

import (
	"fmt"
	"math"
	"sort"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	neuralnetwork "github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

func sortParamArray(paramArray []map[string]interface{}) {
	tofloat := func(x interface{}) float64 {
		switch xv := x.(type) {
		case int:
			return float64(xv)
		case float64:
			return xv
		default:
			return math.NaN()
		}
	}
	cmp := func(a, b interface{}) int {
		va, vb := tofloat(a), tofloat(b)
		if va < vb {
			return -1
		} else if va > vb {
			return 1
		}
		return 0
	}
	less := func(i, j int) bool {
		return cmp(paramArray[i]["a"], paramArray[j]["a"]) < 0 || (cmp(paramArray[i]["a"], paramArray[j]["a"]) == 0 && cmp(paramArray[i]["b"], paramArray[j]["b"]) < 0)
	}
	sort.Slice(paramArray, less)

}
func ExampleParameterGrid() {
	paramArray := ParameterGrid(map[string][]interface{}{"a": {1, 2, 3}, "b": {10, 11}})
	sortParamArray(paramArray)
	for _, m := range paramArray {
		fmt.Println(m["a"], m["b"])
	}
	// Output:
	// 1 10
	// 1 11
	// 2 10
	// 2 11
	// 3 10
	// 3 11

}

func _ExampleGridSearchCV() {
	RandomState := base.NewLockedSource(7)
	ds := datasets.LoadBoston()
	X, Y := preprocessing.NewStandardScaler().FitTransform(ds.X, ds.Y)

	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "relu", "adam", 1e-4)
	mlp.RandomState = RandomState
	mlp.Shuffle = false
	mlp.BatchSize = 20
	mlp.LearningRateInit = .005
	mlp.MaxIter = 100
	scorer := func(Y, Ypred *mat.Dense) float64 {
		return metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
	}
	var gscv *GridSearchCV
	gscv = &GridSearchCV{
		Estimator: mlp,
		ParamGrid: map[string][]interface{}{
			"Alpha":       {2e-4, 5e-4, 1e-3},
			"WeightDecay": {.0002, .0001, 0},
		},
		Scorer:             scorer,
		LowerScoreIsBetter: true,
		CV:                 &KFold{NSplits: 3, RandomState: RandomState, Shuffle: true},
		Verbose:            true,
		NJobs:              -1}
	gscv.Fit(X, Y)
	fmt.Println("Alpha", gscv.BestParams["Alpha"])
	fmt.Println("WeightDecay", gscv.BestParams["WeightDecay"])
	// Output:
	// Alpha 0.0002
	// WeightDecay 0.0001
}
