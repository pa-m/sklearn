package pipeline

import (
	"fmt"

	"github.com/pa-m/sklearn/base"

	"github.com/pa-m/sklearn/datasets"
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
	// or equivalent:
	pl = NewPipeline(NamedStep{"scaler", scaler}, NamedStep{"pca", pca}, NamedStep{"poly", poly}, NamedStep{"mlp", m})
	// pipeline is clonable
	pl = pl.PredicterClone().(*Pipeline)
	// pipeline is classifier if last step is a classifier
	if !pl.IsClassifier() {
		fmt.Println("shouldn't happen")
	}

	pl.Fit(ds.X, ds.Y)
	nSamples, _ := ds.X.Dims()
	_, nOutputs := ds.Y.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)

	pl.Predict(ds.X, Ypred)
	accuracy := pl.Score(ds.X, ds.Y)
	fmt.Println("accuracy>0.999 ?", accuracy > 0.999)
	if accuracy <= .999 {
		fmt.Println("accuracy:", accuracy)
	}
	// pipeline is a Transformer too
	_, _ = pl.FitTransform(ds.X, ds.Y)

	// Output:
	// accuracy>0.999 ? true

}
