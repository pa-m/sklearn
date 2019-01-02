package modelselection

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// KFold ...
type KFold struct {
	NSplits     int
	Shuffle     bool
	RandomState *rand.Source
}

var (
	_ Splitter = &KFold{}
)

// Splitter is the interface for splitters like KFold
type Splitter interface {
	Split(X, Y *mat.Dense) (ch chan Split)
	GetNSplits(X, Y *mat.Dense) int
}

// Split ...
type Split struct{ TrainIndex, TestIndex []int }

// Split generate Split structs
func (splitter *KFold) Split(X, Y *mat.Dense) (ch chan Split) {
	if splitter.NSplits <= 0 {
		splitter.NSplits = 3
	}
	NSamples, _ := X.Dims()

	Shuffle, intn := rand.Shuffle, rand.Intn
	if splitter.RandomState != nil {
		r := rand.New(*splitter.RandomState)
		Shuffle, intn = r.Shuffle, r.Intn

	}

	ch = make(chan Split)
	go func() {
		for isplit := 0; isplit < splitter.NSplits; isplit++ {
			NTest := NSamples / splitter.NSplits
			// The first n_samples % n_splits folds have size n_samples // n_splits + 1, other folds have size n_samples // n_splits, where n_samples is the number of samples.
			if isplit < NSamples%splitter.NSplits {
				NTest++
			}
			a := make([]int, NSamples)

			for i := range a {
				a[i] = i
			}
			aSwap := func(i, j int) { a[i], a[j] = a[j], a[i] }
			if splitter.Shuffle {
				Shuffle(len(a), aSwap)
			} else {
				start := intn(NSamples)
				for i := 0; i < NTest; i++ {
					aSwap((start+i)%NSamples, NSamples-NTest+i)
				}
			}
			sp := Split{
				TrainIndex: a[:NSamples-NTest],
				TestIndex:  a[NSamples-NTest:],
			}

			ch <- sp
		}
		close(ch)
	}()
	return ch
}

// GetNSplits for KFold
func (splitter *KFold) GetNSplits(X, Y *mat.Dense) int {
	return splitter.NSplits
}
