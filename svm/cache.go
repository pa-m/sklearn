package svm

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

func cachedKernel(X *mat.Dense, CacheSize uint, KernelFunction func(X1, X2 []float64) float64) func(i, j int) float64 {
	m, _ := X.Dims()
	type DiagEntry struct {
		bool    //presence in cache
		float64 // cached value
	}
	type KcacheEntry struct {
		uint64  // time last use
		float64 // cached value
	}
	KcacheCap := (CacheSize << 20) / uint(unsafe.Sizeof(KcacheEntry{}))
	KcacheDiag := make([]DiagEntry, m)
	KcacheNDiag := map[[2]int]*KcacheEntry{}
	Ktime := uint64(0)

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
		if e, ok := KcacheNDiag[[2]int{i, j}]; ok {
			Ktime++
			e.uint64 = Ktime
			return e.float64
		}

		if uint(len(KcacheNDiag)) == KcacheCap {
			var oldt = Ktime
			var old [2]int
			for ij, e := range KcacheNDiag {
				if e.uint64 < oldt {
					oldt = e.uint64
					old = ij
				}
			}
			delete(KcacheNDiag, old)
		}
		Ktime++
		e := &KcacheEntry{Ktime, KernelFunction(X.RawRowView(i), X.RawRowView(j))}
		KcacheNDiag[[2]int{i, j}] = e
		return e.float64
	}

}
