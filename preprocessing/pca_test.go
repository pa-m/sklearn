package preprocessing

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func ExamplePCA() {
	X := mat.NewDense(6, 2, []float64{-1., -1., -2., -1., -3., -2., 1., 1., 2., 1., 3., 2.})
	pca := NewPCA()
	pca.Fit(X, nil)
	Xp, _ := pca.Transform(X, nil)
	fmt.Printf("explained  : %.3f\n", pca.ExplainedVarianceRatio)
	fmt.Printf("Svalues    : %.3f\n", pca.SingularValues)
	fmt.Printf("transformed: %.3f\n", Xp.RawMatrix().Data)
	X2, _ := pca.InverseTransform(Xp, nil)
	fmt.Printf("inversed   : %.3f\n", X2.RawMatrix().Data)
	//expected:=[-1.383405778728807 0.293578697080941
	// -2.221898016633681 -0.2513348437429921
	// -3.605303795362488 0.04224385333794878
	// 1.383405778728807 -0.293578697080941
	// 2.221898016633681 0.2513348437429921
	// 3.605303795362488 -0.04224385333794878]

	// Output:
	// explained  : [0.992 0.008]
	// Svalues    : [6.301 0.550]
	// transformed: [-1.383 0.294 -2.222 -0.251 -3.605 0.042 1.383 -0.294 2.222 0.251 3.605 -0.042]
	// inversed   : [-1.000 -1.000 -2.000 -1.000 -3.000 -2.000 1.000 1.000 2.000 1.000 3.000 2.000]

}
