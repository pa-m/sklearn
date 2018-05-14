package base

import (
	"runtime"
	"sync"
)

// Parallelize split execution over NSamples across threads
func Parallelize(threads, NSamples int, f func(th, start, end int)) {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	if threads == 1 {
		f(0, 0, NSamples)
	} else {
		sampPerTh := (NSamples + threads - 1) / threads
		wg := &sync.WaitGroup{}

		for th := 0; th < threads; th++ {
			start, end := th*sampPerTh, (th+1)*sampPerTh
			if end > NSamples {
				end = NSamples
				if end <= start {
					break
				}
			}
			wg.Add(1)
			go func(th int) { f(th, start, end); wg.Done() }(th)
		}
		wg.Wait()
	}
}
