package step

import "context"

// ProviderRequest is the provider-agnostic generation input.
type ProviderRequest struct {
	SystemPrompt string
	History      []Message
	Tools        []ToolSpec
}

// ProviderUpdate is the union-style streaming output from providers.
// It is either a ProviderDeltaUpdate or a ProviderMessageUpdate.
type ProviderUpdate interface {
	isProviderUpdate()
}

// ProviderDeltaUpdate streams incremental deltas.
type ProviderDeltaUpdate struct {
	Delta MessageDelta
}

func (ProviderDeltaUpdate) isProviderUpdate() {}

// ProviderMessageUpdate emits the final assistant message.
type ProviderMessageUpdate struct {
	Message AssistantMessage
}

func (ProviderMessageUpdate) isProviderUpdate() {}

// ProviderStream is the unified provider stream.
// It emits ProviderUpdate values until io.EOF.
type ProviderStream interface {
	Next(ctx context.Context) (ProviderUpdate, error)
	Close() error
}

// Provider is the unified interface implemented by providers.
type Provider interface {
	Stream(ctx context.Context, req ProviderRequest) (ProviderStream, error)
}
