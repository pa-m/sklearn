package linearmodel

import "github.com/pa-m/sklearn/base"

// Ridge is an alias for RegularizedRegression
type Ridge = RegularizedRegression

// NewRidge creates a *RegularizedRegression with Alpha=1. and L1Ratio=0
func NewRidge() *RegularizedRegression {
	regr := &RegularizedRegression{}
	regr.FitIntercept = true
	regr.Tol = 1e-3
	regr.Alpha = 1.
	regr.L1Ratio = 0.
	//regr.MaxIter = 1000
	return regr
}

// PredicterClone for Ridge
func (m *Ridge) PredicterClone() base.Predicter {
	clone := *m
	return &clone
}
