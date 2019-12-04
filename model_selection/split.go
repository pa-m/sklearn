package modelselection

import (
	"math"

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
	SplitterClone() Splitter
}

// Split ...
type Split struct{ TrainIndex, TestIndex []int }

// SplitterClone ...
func (splitter *KFold) SplitterClone() Splitter {
	if splitter == nil {
		return nil
	}
	clone := *splitter
	if sourceCloner, ok := clone.RandomState.(base.SourceCloner); ok && sourceCloner != base.SourceCloner(nil) {
		clone.RandomState = sourceCloner.SourceClone()
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
		} else {
			rndShuffle = rand.New(splitter.RandomState).Shuffle
		}
		if intner, ok := splitter.RandomState.(Intner); ok {
			rndIntn = intner.Intn
		} else {
			rndIntn = rand.New(splitter.RandomState).Intn
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

// TrainTestSplit splits X and Y into test set and train set
// testsize must be between 0 and 1
// it produce same sets than scikit-learn
func TrainTestSplit(X, Y mat.Matrix, testsize float64, randomstate uint64) (Xtrain, Xtest, ytrain, ytest *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	_, NOutputs := Y.Dims()
	var testlen int
	if testsize > 1 {
		testlen = int(math.Ceil(math.Min(float64(NSamples), testsize)))
	} else {
		testlen = int(math.Ceil(float64(NSamples) * testsize))
	}
	Xtest = mat.NewDense(testlen, NFeatures, nil)
	ytest = mat.NewDense(testlen, NOutputs, nil)
	Xtrain = mat.NewDense(NSamples-testlen, NFeatures, nil)
	ytrain = mat.NewDense(NSamples-testlen, NOutputs, nil)
	src := base.NewLockedSource(randomstate)

	var ind []int
	src.WithLock(func(src base.Source) {
		permer, ok := src.(base.Permer)
		if !ok {
			panic("Source does not implement Perm")
		}
		{
			ind = permer.Perm(NSamples)
		}

	})
	for i := 0; i < NSamples; i++ {
		j := ind[i]
		if i < testlen {
			mat.Row(Xtest.RawRowView(i), j, X)
			mat.Row(ytest.RawRowView(i), j, Y)
		} else {
			mat.Row(Xtrain.RawRowView(i-testlen), j, X)
			mat.Row(ytrain.RawRowView(i-testlen), j, Y)
		}
	}
	return
}
