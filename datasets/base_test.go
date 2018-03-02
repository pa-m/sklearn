package datasets

import (

	//"fmt"
	"fmt"
	"pa-m/sklearn/base"
	"testing"
)

func TestLoadIris(t *testing.T) {
	ds := LoadIris()
	//fmt.Printf("iris data:%#v\n", ds)
	if ds == nil {
		t.Fail()
	}
}

var matstr = base.MatStr

func ExampleIrisGetMatrices() {
	X, Y := IrisGetMatrices(LoadIris())
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
