package step

import "context"

// GenerateRequest is the provider-agnostic generation input.
type GenerateRequest struct {
	SystemPrompt string
	History      []Message
	Tools        []ToolSpec
}

// GenerateResult is the provider-agnostic generation output.
type GenerateResult struct {
	Message    Message
	Usage      *Usage
	StopReason StopReason
}

// AssistantStream streams assistant events.
type AssistantStream interface {
	Next(ctx context.Context) (AssistantEvent, error)
	Result() (*GenerateResult, error)
	Close() error
}

// ChatProvider is the unified interface implemented by providers.
type ChatProvider interface {
	GenerateStream(ctx context.Context, req GenerateRequest) (AssistantStream, error)
}
