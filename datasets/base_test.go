package datasets

import (

	//"fmt"
	"fmt"
	"testing"

	"github.com/pa-m/sklearn/base"

	"gonum.org/v1/gonum/mat"
)

func TestLoadIris(t *testing.T) {
	ds := LoadIris()
	//fmt.Printf("iris data:%#v\n", ds)
	if ds == nil {
		t.Fail()
	}
}

var matstr = base.MatStr

func ExampleLoadIris() {
	ds := LoadIris()
	X, Y := ds.X, ds.Y
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 4), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 150,4 Y 150,1
	// 5.1	3.5	1.4	0.2	0
	// 4.9	3	1.4	0.2	0
	// 4.7	3.2	1.3	0.2	0
}

func ExampleLoadBreastCancer() {
	ds := LoadBreastCancer()
	X, Y := ds.X, ds.Y
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 4), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 569,30 Y 569,1
	// 17.99	10.38	122.8	1001	0
	// 20.57	17.77	132.9	1326	0
	// 19.69	21.25	130	1203	0
}

func ExampleLoadDiabetes() {
	ds := LoadDiabetes()
	X, Y := ds.X, ds.Y

	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 4), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 442,10 Y 442,1
	// 0.0380759064334241	0.0506801187398187	0.0616962065186885	0.0218723549949558	151
	// -0.00188201652779104	-0.044641636506989	-0.0514740612388061	-0.0263278347173518	75
	// 0.0852989062966783	0.0506801187398187	0.0444512133365941	-0.00567061055493425	141
}

func ExampleLoadBoston() {
	X, Y := LoadBoston().GetXY()
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 4), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 506,13 Y 506,1
	// 0.00632	18	2.31	0	24
	// 0.02731	0	7.07	0	21.6
	// 0.02729	0	7.07	0	34.7
}

func ExampleLoadExamScore() {
	X, Y := LoadExamScore()
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 2), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 100,2 Y 100,1
	// 34.62365962451697	78.0246928153624	0
	// 30.28671076822607	43.89499752400101	0
	// 35.84740876993872	72.90219802708364	0

}

func ExampleLoadMicroChipTest() {
	X, Y := LoadMicroChipTest()
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, 2), Y.Slice(0, 3, 0, 1)))
	// Output:
	// X 118,2 Y 118,1
	// 0.051267	0.69956	1
	// -0.092742	0.68494	1
	// -0.21371	0.69225	1
}

func ExampleLoadMnist() {
	X, Y := LoadMnist()
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Printf("%.6f %.6f\n", mat.Sum(X), mat.Sum(Y))
	// Output:
	// X 5000,400 Y 5000,1
	// 262678.260160 27500.000000
}

func ExampleLoadMnistWeights() {
	Theta1, Theta2 := LoadMnistWeights()
	xr, xc := Theta1.Dims()
	yr, yc := Theta2.Dims()
	fmt.Printf("Theta1 %d,%d Theta2 %d,%d\n", xr, xc, yr, yc)
	fmt.Printf("%.6f %.6f\n", mat.Sum(Theta1), mat.Sum(Theta2))
	// Output:
	// Theta1 25,401 Theta2 10,26
	// 9.242644 -100.083444
}

func ExampleLoadInternationalAirlinesPassengers() {
	Y := LoadInternationalAirlinesPassengers()
	fmt.Println(Y.Dims())
	// Output:
	// 144 1
}

func ExampleLoadWine() {
	X, Y := LoadWine().GetXY()
	xr, xc := X.Dims()
	yr, yc := Y.Dims()
	fmt.Printf("X %d,%d Y %d,%d\n", xr, xc, yr, yc)
	fmt.Println(matstr(X.Slice(0, 3, 0, xc), Y.Slice(0, 3, 0, yc)))
	// Output:
	//X 178,13 Y 178,1
	//14.23	1.71	2.43	15.6	127	2.8	3.06	0.28	2.29	5.64	1.04	3.92	1065	0
	//13.2	1.78	2.14	11.2	100	2.65	2.76	0.26	1.28	4.38	1.05	3.4	1050	0
	//13.16	2.36	2.67	18.6	101	2.8	3.24	0.3	2.81	5.68	1.03	3.17	1185	0

}
