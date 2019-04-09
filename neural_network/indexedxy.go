package neuralnetwork

import "sort"

// indexedXY implements sort.Slice to be used in Shuffle and sort.Sort
type indexedXY struct {
	idx, X, Y sort.Interface
}

func (s indexedXY) Len() int           { return s.idx.Len() }
func (s indexedXY) Less(i, j int) bool { return s.idx.Less(i, j) }
func (s indexedXY) Swap(i, j int)      { s.idx.Swap(i, j); s.X.Swap(i, j); s.Y.Swap(i, j) }
