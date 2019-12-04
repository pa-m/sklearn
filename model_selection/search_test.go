package modelselection

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	neuralnetwork "github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/preprocessing"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func TestGridSearchCV_PredicterClone(t *testing.T) {
	RandomState := base.NewLockedSource(7)

	mlp := neuralnetwork.NewMLPRegressor([]int{}, "relu", "adam", 1e-4)
	m := &GridSearchCV{
		Estimator: mlp,
		ParamGrid: map[string][]interface{}{
			"Alpha":       {2e-4, 5e-4, 1e-3},
			"WeightDecay": {.0002, .0001, 0},
		},
		Scorer:             mlp.Score,
		LowerScoreIsBetter: false,
		CV:                 &KFold{NSplits: 3, RandomState: RandomState, Shuffle: true},
		Verbose:            true,
		NJobs:              -1}
	clone := m.PredicterClone()
	expected, actual := fmt.Sprintf("%+v", m), fmt.Sprintf("%+v", clone)
	if m == clone {
		t.Fail()
	}
	if actual != expected {
		t.Errorf("\nexpected: %s\ngot     : %s", expected, actual)
	}
}
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

func chkRandomState(rs rand.Source) {
	expected := fmt.Sprintf("%+v\n", base.NewSource(7))
	got := fmt.Sprintf("%+v\n", rs)
	if got != expected {
		panic(fmt.Errorf("wrong random state\nexpected:%s\n%s\ngot     :%s\n%s", expected, "", got, ""))
	}
}

func ExampleGridSearchCV() {
	RandomState := base.NewLockedSource(7)
	ds := datasets.LoadBoston()
	X, Y := preprocessing.NewStandardScaler().FitTransform(ds.X, ds.Y)

	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "relu", "adam", 1e-4)
	mlp.RandomState = RandomState
	mlp.Shuffle = false
	mlp.BatchSize = 20
	mlp.LearningRateInit = .005
	mlp.MaxIter = 100

	scorer := func(Y, Ypred mat.Matrix) float64 {
		return metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
	}
	gscv := &GridSearchCV{
		Estimator: mlp,
		ParamGrid: map[string][]interface{}{
			"Alpha":       {1e-4, 2e-4, 5e-4, 1e-3},
			"WeightDecay": {1e-4, 1e-5, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 0},
		},
		Scorer:             scorer,
		LowerScoreIsBetter: true,
		// CV:                 &KFold{NSplits: 3, RandomState: RandomState, Shuffle: true},
		Verbose: true,
		NJobs:   -1}
	gscv.Fit(X, Y)
	fmt.Println("Alpha", gscv.BestParams["Alpha"])
	fmt.Println("WeightDecay", gscv.BestParams["WeightDecay"])
	// fmt.Println(gscv.CVResults["score"])

	// Output:
	//Alpha 0.0001
	//WeightDecay 5e-08

}

func Test_setParam(t *testing.T) {
	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "relu", "adam", 1e-4)
	setParam(mlp, "activation", "logistic")
	setParam(mlp, "Alpha", 1)
	str, ok := getParam(mlp, "activation")
	if !ok {
		t.Fail()
	}
	if str.(string) != "logistic" {
		t.Fail()
	}
	alpha, ok2 := getParam(mlp, "alpha")
	if !ok2 {
		t.Fail()
	}
	if alpha.(float64) != 1 {
		t.Fail()
	}
}
