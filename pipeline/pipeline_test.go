package pipeline

import (
	"fmt"

	"github.com/pa-m/sklearn/base"
	"github.com/pa-m/sklearn/preprocessing"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	nn "github.com/pa-m/sklearn/neural_network"
	"gonum.org/v1/gonum/mat"
)

func _ExamplePipeline() {
	ds := datasets.LoadBreastCancer()
	fmt.Println("Dims", base.MatDimsString(ds.X, ds.Y))

	chkTransformer := func(t preprocessing.Transformer) {}

	scaler := preprocessing.NewStandardScaler()
	chkTransformer(scaler)

	pca := preprocessing.NewPCA()
	chkTransformer(pca)

	poly := preprocessing.NewPolynomialFeatures(2)
	poly.IncludeBias = false
	chkTransformer(poly)

	m := nn.NewMLPClassifier([]int{}, "relu", "adam", 0.)
	m.Loss = "cross-entropy"

	m.Epochs = 300

	pl := MakePipeline(scaler, pca, poly, m)

	pl.Fit(ds.X, ds.Y)
	nSamples, _ := ds.X.Dims()
	_, nOutputs := ds.Y.Dims()
	Ypred := mat.NewDense(nSamples, nOutputs, nil)
	pl.Predict(ds.X, Ypred)
	accuracy := metrics.AccuracyScore(ds.Y, Ypred, true, nil)
	fmt.Println("accuracy>0.99 ?", accuracy > 0.99)
	if accuracy <= 0.99 {
		fmt.Println("accuracy:", accuracy)
	}
	// Output:
	// Dims  569,30 569,1
	// accuracy>0.99 ? true

}
