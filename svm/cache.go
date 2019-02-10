package svm

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

func cachedKernel(X *mat.Dense, CacheSize uint, KernelFunction func(X1, X2 []float64) float64) func(i, j int) float64 {
	m, _ := X.Dims()
	type DiagEntry struct {
		bool
		float64
	}
	type KcacheEntry struct {
		i, j int
		float64
	}
	KcacheEntrySize := uint(unsafe.Sizeof(KcacheEntry{}))

	KcacheDiag := make([]DiagEntry, m, m)
	KcacheNDiag := make([]KcacheEntry, 0, CacheSize<<20/KcacheEntrySize)

	return func(i, j int) float64 {
		if i == j {
			e := &KcacheDiag[i]
			if !e.bool {
				e.float64 = KernelFunction(X.RawRowView(i), X.RawRowView(i))
				e.bool = true
			}
			return e.float64
		}
		if i > j {
			i, j = j, i
		}
		for off, e := range KcacheNDiag {
			if e.i == i && e.j == j {
				copy(KcacheNDiag[1:off+1], KcacheNDiag[0:off])
				KcacheNDiag[0] = e
				return e.float64
			}
		}
		if uint(len(KcacheNDiag))*KcacheEntrySize >= CacheSize {
			KcacheNDiag = KcacheNDiag[0 : CacheSize/KcacheEntrySize]
		}
		e := KcacheEntry{i, j, KernelFunction(X.RawRowView(i), X.RawRowView(j))}
		KcacheNDiag = append([]KcacheEntry{e}, KcacheNDiag...)
		return e.float64
	}

}
