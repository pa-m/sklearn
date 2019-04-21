package neuralnetwork

import (
	"fmt"
	"testing"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/datasets"
)

func ExampleBaseMultilayerPerceptron32_Fit_mnist() {
	X, Y := datasets.LoadMnist()
	mlp := NewBaseMultilayerPerceptron32()
	mlp.HiddenLayerSizes = []int{25}
	mlp.MaxIter = 200
	mlp.RandomState = base.NewLockedSource(7)
	mlp.Shuffle = true
	mlp.BatchNormalize = true
	expectedMinAccuracy := .988

	if testing.Short() {
		mlp.MaxIter = 20
		expectedMinAccuracy = .93
	}
	mlp.Fit(X, Y)
	accuracy := mlp.Score(X, Y)
	if accuracy < expectedMinAccuracy {
		fmt.Println("accuracy", accuracy)
	} else {
		fmt.Println("ok")
	}
	// Output:
	// ok
}
