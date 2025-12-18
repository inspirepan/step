package step

import "errors"

var (
	ErrNoProvider   = errors.New("step: provider is required")
	ErrToolNotFound = errors.New("step: tool not found")
)
