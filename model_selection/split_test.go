package modelselection

import (
	"fmt"

	"github.com/pa-m/sklearn/datasets"

	"github.com/pa-m/sklearn/base"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func ExampleKFold() {
	randomState := rand.New(base.NewLockedSource(7))
	X := mat.NewDense(6, 1, []float64{1, 2, 3, 4, 5, 6})
	subtest := func(shuffle bool) {
		fmt.Println("shuffle", shuffle)
		kf := &KFold{NSplits: 3, Shuffle: shuffle, RandomState: randomState}
		for sp := range kf.Split(X, nil) {
			fmt.Printf("%#v\n", sp)
		}

	}
	subtest(false)
	subtest(true)
	// Output:
	// shuffle false
	// modelselection.Split{TrainIndex:[]int{0, 1, 2, 3}, TestIndex:[]int{4, 5}}
	// modelselection.Split{TrainIndex:[]int{4, 5, 2, 3}, TestIndex:[]int{0, 1}}
	// modelselection.Split{TrainIndex:[]int{0, 4, 5, 3}, TestIndex:[]int{1, 2}}
	// shuffle true
	// modelselection.Split{TrainIndex:[]int{5, 0, 2, 3}, TestIndex:[]int{4, 1}}
	// modelselection.Split{TrainIndex:[]int{5, 3, 2, 0}, TestIndex:[]int{1, 4}}
	// modelselection.Split{TrainIndex:[]int{2, 4, 1, 0}, TestIndex:[]int{5, 3}}
}

func perm(r base.Intner, n int) []int {

	m := make([]int, n)
	// In the following loop, the iteration when i=0 always swaps m[0] with m[0].
	// A change to remove this useless iteration is to assign 1 to i in the init
	// statement. But Perm also effects r. Making this change will affect
	// the final state of r. So this change can't be made for compatibility
	// reasons for Go 1.
	for i := 0; i < n; i++ {
		j := r.Intn(i + 1)
		m[i] = m[j]
		m[j] = i
	}
	return m

}

func _ExampleTrainTestSplit() {

	features, target := datasets.LoadWine().GetXY()
	RandomState := uint64(42)
	_, _, Ytrain, Ytest := TrainTestSplit(features, target, .30, RandomState)
	Ntrain, _ := Ytrain.Dims()
	ytrain := make([]float64, Ntrain)
	mat.Col(ytrain, 0, Ytrain)
	fmt.Println(ytrain[:8])
	Ntest, _ := Ytest.Dims()
	ytest := make([]float64, Ntest)
	mat.Col(ytest, 0, Ytest)
	fmt.Println(ytest[:8])

	// Output:
	//[2 1 1 0 1 0 2 1]
	//[0 0 2 0 1 0 1 2]
}

func ExampleTrainTestSplit() {
	/*
	   >>> import numpy as np
	   >>> from sklearn.model_selection import train_test_split
	   >>> X, y = np.arange(10).reshape((5, 2)), range(5)
	   >>> X_train, X_test, y_train, y_test = train_test_split(
	   ...     X, y, test_size=0.33, random_state=42)
	   ...
	   >>> X_train
	   array([[4, 5],
	          [0, 1],
	          [6, 7]])
	   >>> y_train
	   [2, 0, 3]
	   >>> X_test
	   array([[2, 3],
	          [8, 9]])
	   >>> y_test
	   [1, 4]

	*/
	X := mat.NewDense(5, 2, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	Y := mat.NewDense(5, 1, []float64{0, 1, 2, 3, 4})
	RandomState := uint64(42)
	Xtrain, Xtest, Ytrain, Ytest := TrainTestSplit(X, Y, .33, RandomState)
	fmt.Printf("X_train:\n%g\n", mat.Formatted(Xtrain))
	fmt.Printf("Y_train:\n%g\n", mat.Formatted(Ytrain))
	fmt.Printf("X_test:\n%g\n", mat.Formatted(Xtest))
	fmt.Printf("Y_test:\n%g\n", mat.Formatted(Ytest))

	// Output:
	//X_train:
	//⎡4  5⎤
	//⎢0  1⎥
	//⎣6  7⎦
	//Y_train:
	//⎡2⎤
	//⎢0⎥
	//⎣3⎦
	//X_test:
	//⎡2  3⎤
	//⎣8  9⎦
	//Y_test:
	//⎡1⎤
	//⎣4⎦

}
