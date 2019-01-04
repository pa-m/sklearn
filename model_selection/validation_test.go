package modelselection

import (
	"fmt"
	"math"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/pipeline"
	"github.com/pa-m/sklearn/preprocessing"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func ExampleCrossValidate() {
	randomState := int64(5)

	ds := datasets.LoadBoston()
	X, Y := ds.X, ds.Y
	mlp := neuralnetwork.NewMLPRegressor([]int{20}, "identity", "adam", 1e-4)
	mlp.RandomState = &randomState
	mlp.Shuffle = false
	mlp.MiniBatchSize = 5
	mlp.WeightDecay = .1

	mlp.Epochs = 50
	m := pipeline.NewPipeline(
		pipeline.NamedStep{Name: "standardize", Step: preprocessing.NewStandardScaler()},
		pipeline.NamedStep{Name: "mlpregressor", Step: mlp},
	)
	scorer := func(Y, Ypred *mat.Dense) float64 {
		e := metrics.MeanSquaredError(Y, Ypred, nil, "").At(0, 0)
		return e
	}
	mean := func(x []float64) float64 { return floats.Sum(x) / float64(len(x)) }
	res := CrossValidate(m, X, Y,
		nil,
		scorer,
		&KFold{NSplits: 10, Shuffle: true, RandomState: &randomState}, 10)
	fmt.Println(math.Sqrt(mean(res.TestScore)) < 6.)
	// Output:
	// true
}
