package pipeline

import (
	"fmt"

	"github.com/pa-m/sklearn/base"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	nn "github.com/pa-m/sklearn/neural_network"
	"github.com/pa-m/sklearn/preprocessing"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func ExamplePipeline() {
	randomState := rand.New(base.NewLockedSource(7))

	ds := datasets.LoadBreastCancer()

	scaler := preprocessing.NewStandardScaler()

	pca := preprocessing.NewPCA()
	pca.MinVarianceRatio = 0.995

	poly := preprocessing.NewPolynomialFeatures(2)
	poly.IncludeBias = false

	m := nn.NewMLPClassifier([]int{}, "relu", "adam", 0)
	m.RandomState = randomState
	m.MaxIter = 300
	m.LearningRateInit = .02
	m.WeightDecay = .001

	pl := MakePipeline(scaler, pca, poly, m)

	pl.Fit(ds.X, ds.Y)
	nSamples, _ := ds.X.Dims()
	_, nOutputs := ds.Y.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)

	pl.Predict(ds.X, Ypred)
	accuracy := metrics.AccuracyScore(ds.Y, Ypred, true, nil)
	fmt.Println("accuracy>0.999 ?", accuracy > 0.999)
	if accuracy <= .999 {
		fmt.Println("accuracy:", accuracy)
	}
	// Output:
	// accuracy>0.999 ? true

}
