package modelselection

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	neuralnetwork "github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/mat"
)

func ExampleParameterGrid() {
	arrOfMap := ParameterGrid(map[string][]interface{}{"a": {1, 2, 3}, "b": {10, 11}})
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
		return cmp(arrOfMap[i]["a"], arrOfMap[j]["a"]) < 0 || (cmp(arrOfMap[i]["a"], arrOfMap[j]["a"]) == 0 && cmp(arrOfMap[i]["b"], arrOfMap[j]["b"]) < 0)
	}
	sort.Slice(arrOfMap, less)
	for _, m := range arrOfMap {
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

func ExampleGridSearchCV() {
	ds := datasets.LoadBoston()
	X, Y := preprocessing.NewStandardScaler().FitTransform(ds.X, ds.Y)

	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "relu", "adam", 1e-4)
	mlp.Shuffle = false
	mlp.MiniBatchSize = 5
	mlp.Epochs = 100
	scorer := func(Y, Ypred *mat.Dense) float64 {
		e := -metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
		return e
	}
	RandomState := rand.NewSource(7)
	gscv := &GridSearchCV{
		Estimator: mlp,
		ParamGrid: map[string][]interface{}{"Alpha": {0, 1e-4}, "WeightDecay": {0, 0.1}},
		Scorer:    scorer,
		CV:        &KFold{NSplits: 3, RandomState: &RandomState},
		Verbose:   true,
		NJobs:     -1}
	gscv.Fit(X, Y)
	fmt.Println(gscv.BestParams["Alpha"], gscv.BestParams["WeightDecay"])
	//fmt.Printf("%#v\n", gscv.CVResults)

	// Output:
	// 0.0001 0.1
}
