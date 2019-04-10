package neuralnetwork

import (
	"fmt"
	"testing"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/preprocessing"
)

func ExampleBaseMultilayerPerceptron32_Fit_mnist() {
	X, Y := datasets.LoadMnist()
	_, Ybin := preprocessing.NewLabelBinarizer(0, 1).FitTransform(nil, Y)
	mlp := NewBaseMultilayerPerceptron32()
	mlp.HiddenLayerSizes = []int{25}
	mlp.LossFuncName = "binary_log_loss"
	mlp.MaxIter = 200
	mlp.Shuffle = true
	mlp.BatchNormalize = true
	minAcc := .988

	if testing.Short() {
		mlp.MaxIter = 20
		minAcc = .93
	}
	mlp.Fit(X, Ybin)
	accuracy := mlp.Score(X, Ybin)
	if accuracy < minAcc {
		fmt.Println("accuracy", accuracy)
	} else {
		fmt.Println("ok")
	}
	// Output:
	// ok
}
