package base

import (
	"sync"

	"github.com/pa-m/randomkit"
	"golang.org/x/exp/rand"
)

// A Source represents a source of uniformly-distributed
// pseudo-random int64 values in the range [0, 1<<64).
type Source = rand.Source

// SourceCloner is an "golang.org/x/exp/rand".Source with a Clone method
type SourceCloner interface {
	SourceClone() Source
}

// RandomState represents a bit more than random_state pythonic attribute. it's not only a seed but a source with a state as it's name states
type RandomState = Source

// NewSource returns a new pseudo-random Source seeded with the given value.
func NewSource(seed uint64) *randomkit.RKState {
	var rng randomkit.RKState
	rng.Seed(seed)
	return &rng
}

// LockedSource is an implementation of Source that is concurrency-safe.
// It is just a standard Source with its operations protected by a sync.Mutex.
type LockedSource struct {
	lk  sync.Mutex
	src *randomkit.RKState
}

// WithLock executes f while s is locked
func (s *LockedSource) WithLock(f func(Source)) {
	s.lk.Lock()
	f(s.src)
	s.lk.Unlock()
}

// Uint64 ...
func (s *LockedSource) Uint64() (n uint64) {
	s.lk.Lock()
	n = s.src.Uint64()
	s.lk.Unlock()
	return
}

// Seed ...
func (s *LockedSource) Seed(seed uint64) {
	s.lk.Lock()
	s.src.Seed(seed)
	s.lk.Unlock()
}

// SourceClone ...
func (s *LockedSource) SourceClone() Source {
	return &LockedSource{src: s.src.SourceClone().(*randomkit.RKState)}
}

// NewLockedSource returns a rand.Source safe for concurrent access
func NewLockedSource(seed uint64) *LockedSource {
	var s LockedSource
	s.src = NewSource(seed)
	return &s
}

// Float64er is implemented by a random source having a method Float64() float64
type Float64er interface {
	Float64() float64
}

// NormFloat64er is implemented by a random source having a method NormFloat64() float64
type NormFloat64er interface {
	NormFloat64() float64
}

// Intner is implemented by a random source having a method Intn() float64
type Intner interface {
	Intn(int) int
}

// Permer is implemented by a random source having a method Perm(int) []int
type Permer interface {
	Perm(int) []int
}

// Shuffler  is implemented by a random source having a method Shuffle(int,func(int,int))
type Shuffler interface {
	Shuffler(int, func(int, int))
}
