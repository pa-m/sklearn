package base

import (
	"sync"

	"github.com/pa-m/randomkit"
)

// A Source represents a source of uniformly-distributed
// pseudo-random int64 values in the range [0, 1<<64).
type Source interface {
	Uint64() uint64
	Seed(seed uint64)
}

// RandomState represents a bit more than random_state pythonic attribute. it's not only a seed but a source with a state as it's name states
type RandomState = Source

// NewSource returns a new pseudo-random Source seeded with the given value.
func NewSource(seed uint64) Source {
	var rng randomkit.RKState
	rng.Seed(seed)
	return &rng
}

// LockedSource is an implementation of Source that is concurrency-safe.
// It is just a standard Source with its operations protected by a sync.Mutex.
type LockedSource struct {
	lk  sync.Mutex
	src Source
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

// NewLockedSource returns a rand.Source safe for concurrent access
func NewLockedSource(seed uint64) Source {
	var s LockedSource
	s.src = NewSource(seed)
	return &s
}

// NormFloat64er is implemented by a random source having a method NormFloat64() float64
type NormFloat64er interface {
	NormFloat64() float64
}

// Intner is implemented by a random source having a method Intn}() float64
type Intner interface {
	Intn(int) int
}
