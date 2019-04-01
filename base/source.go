package base

import (
	"sync"

	"golang.org/x/exp/rand"
)

// LockedSource is an implementation of Source that is concurrency-safe.
// It is just a standard Source with its operations protected by a sync.Mutex.
type LockedSource struct {
	lk  sync.Mutex
	src rand.Source
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
func NewLockedSource(seed uint64) rand.Source {
	var s LockedSource
	s.src = rand.NewSource(seed)
	return &s
}
