package modelselection

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/pipeline"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func ExampleCrossValidate() {
	ds := datasets.LoadBoston()
	X, Y := ds.X, ds.Y
	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "identity", "adam", 1e-4)
	mlp.Shuffle = false
	mlp.MiniBatchSize = 5
	mlp.Epochs = 100
	m := pipeline.NewPipeline(
		pipeline.NamedStep{Name: "standardize", Step: preprocessing.NewStandardScaler()},
		pipeline.NamedStep{Name: "mlpregressor", Step: mlp},
	)
	RandomState := rand.NewSource(5)
	scorer := func(Y, Ypred *mat.Dense) float64 {
		e := metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
		return e
	}
	mean := func(x []float64) float64 { return floats.Sum(x) / float64(len(x)) }
	res := CrossValidate(m, X, Y,
		nil,
		scorer,
		&KFold{NSplits: 10, Shuffle: true, RandomState: &RandomState}, 10)
	fmt.Println(math.Sqrt(mean(res.TestScore)) < 25)
	// Output:
	// true
}
