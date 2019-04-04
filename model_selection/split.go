package modelselection

import (
	"github.com/pa-m/sklearn/base"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
)

// RandomState is to init a new random source for reproducibility
type RandomState = rand.Rand

// KFold ...
type KFold struct {
	NSplits     int
	Shuffle     bool
	RandomState base.RandomState
}

var (
	_ Splitter = &KFold{}
)

// Splitter is the interface for splitters like KFold
type Splitter interface {
	Split(X, Y *mat.Dense) (ch chan Split)
	GetNSplits(X, Y *mat.Dense) int
	Clone() Splitter
}

// Split ...
type Split struct{ TrainIndex, TestIndex []int }

// Clone ...
func (splitter *KFold) Clone() Splitter {
	clone := *splitter
	type Cloner interface{ Clone() base.Source }
	if clone.RandomState != nil {
		if cloner, ok := clone.RandomState.(Cloner); ok {
			clone.RandomState = cloner.Clone()
		}
	}
	return &clone
}

// Split generate Split structs
func (splitter *KFold) Split(X, Y *mat.Dense) (ch chan Split) {
	if splitter.NSplits <= 0 {
		splitter.NSplits = 3
	}
	NSamples, _ := X.Dims()

	type Shuffler interface {
		Shuffle(n int, swap func(i, j int))
	}
	type Intner interface{ Intn(int) int }
	var rndShuffle = rand.Shuffle
	var rndIntn = rand.Intn

	if splitter.RandomState != base.Source(nil) {
		if shuffler, ok := splitter.RandomState.(Shuffler); ok {
			rndShuffle = shuffler.Shuffle
		}
		if intner, ok := splitter.RandomState.(Intner); ok {
			rndIntn = intner.Intn
		}
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
				rndShuffle(len(a), aSwap)
			} else {
				start := rndIntn(NSamples)
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
