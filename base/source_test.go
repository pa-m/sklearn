package base

import (
	"golang.org/x/exp/rand"

	"github.com/pa-m/randomkit"
)

var (
	_ = []rand.Source{&randomkit.RKState{}, &LockedSource{}}
	_ = []Float64er{&randomkit.RKState{}}
	_ = []NormFloat64er{&randomkit.RKState{}}
	_ = []SourceCloner{&LockedSource{}}
)
