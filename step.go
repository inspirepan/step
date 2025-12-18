package step

import (
	"context"
)

// StepRequest configures a single agent step.
type StepRequest struct {
	Provider     Provider
	SystemPrompt string
	History      []Message
	Tools        []Tool
}

// StepOption configures optional step behavior.
type StepOption func(*stepConfig)

type stepConfig struct {
	stepEmitter
}

// StepCallbacks provides optional hooks for observing streaming updates.
//
// Callbacks are invoked sequentially in the caller goroutine. Keep them fast.
type StepCallbacks struct {
	OnDelta   func(MessageDelta)
	OnMessage func(Message)
}

// WithCallbacks configures callback hooks.
func WithCallbacks(cb StepCallbacks) StepOption {
	return func(c *stepConfig) {
		if cb.OnDelta != nil {
			c.onDelta = cb.OnDelta
		}
		if cb.OnMessage != nil {
			c.onMessage = cb.OnMessage
		}
	}
}

// WithOnDelta configures a delta hook.
func WithOnDelta(fn func(MessageDelta)) StepOption {
	return func(c *stepConfig) { c.onDelta = fn }
}

// WithOnMessage configures a message hook.
func WithOnMessage(fn func(Message)) StepOption {
	return func(c *stepConfig) { c.onMessage = fn }
}

// StepResult is the sequence of new messages produced by a step.
// It is safe to append to the conversation history.
type StepResult []Message

// Step runs one step synchronously.
func Step(ctx context.Context, req StepRequest, opts ...StepOption) (StepResult, error) {
	var cfg stepConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return runStep(ctx, req, cfg)
}
