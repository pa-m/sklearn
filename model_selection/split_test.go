package modelselection

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func ExampleKFold() {
	X := mat.NewDense(6, 1, []float64{1, 2, 3, 4, 5, 6})
	subtest := func(shuffle bool) {
		RandomState := rand.NewSource(7)
		fmt.Println("shuffle", shuffle)
		kf := &KFold{NSplits: 3, Shuffle: shuffle, RandomState: &RandomState}
		for sp := range kf.Split(X, nil) {
			fmt.Printf("%#v\n", sp)
		}

	}
	subtest(false)
	subtest(true)
	// Output:
	// shuffle false
	// modelselection.Split{TrainIndex:[]int{0, 1, 4, 5}, TestIndex:[]int{2, 3}}
	// modelselection.Split{TrainIndex:[]int{4, 5, 2, 3}, TestIndex:[]int{0, 1}}
	// modelselection.Split{TrainIndex:[]int{0, 1, 2, 4}, TestIndex:[]int{5, 3}}
	// shuffle true
	// modelselection.Split{TrainIndex:[]int{3, 4, 2, 0}, TestIndex:[]int{1, 5}}
	// modelselection.Split{TrainIndex:[]int{2, 3, 5, 4}, TestIndex:[]int{1, 0}}
	// modelselection.Split{TrainIndex:[]int{4, 3, 5, 0}, TestIndex:[]int{1, 2}}

}
