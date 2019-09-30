package modelselection

import (
	"fmt"
	"github.com/pa-m/randomkit"
	"testing"

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
func TestTrainTestSplit(t *testing.T) {
	rs := randomkit.NewRandomkitSource(42)
	NSamples := 178
	ind := make([]int, NSamples)
	for i := range ind {
		ind[i] = i
	}
	permer := rand.New(rs)
	ind = permer.Perm(178)
	fmt.Println(ind)
	// Output:
}
