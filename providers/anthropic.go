package providers

import (
	"context"

	"github.com/inspirepan/step"
)

// AnthropicConfig configures Anthropic Messages API provider.
type AnthropicConfig struct {
	APIKey  string
	BaseURL string

	// Generation options
	MaxOutputTokens *int
	Temperature     *float64

	// Thinking options
	ThinkingEnabled bool
	ThinkingBudget  *int
}

// AnthropicOption is a functional option for Anthropic provider.
type AnthropicOption func(*AnthropicConfig)

// AnthropicWithAPIKey sets the API key.
func AnthropicWithAPIKey(key string) AnthropicOption {
	return func(c *AnthropicConfig) { c.APIKey = key }
}

// AnthropicWithBaseURL sets a custom base URL.
func AnthropicWithBaseURL(url string) AnthropicOption {
	return func(c *AnthropicConfig) { c.BaseURL = url }
}

// AnthropicWithTemperature sets the temperature.
func AnthropicWithTemperature(t float64) AnthropicOption {
	return func(c *AnthropicConfig) { c.Temperature = &t }
}

// AnthropicWithMaxOutputTokens sets the max output tokens.
func AnthropicWithMaxOutputTokens(n int) AnthropicOption {
	return func(c *AnthropicConfig) { c.MaxOutputTokens = &n }
}

// AnthropicWithThinking enables extended thinking.
func AnthropicWithThinking(budget int) AnthropicOption {
	return func(c *AnthropicConfig) {
		c.ThinkingEnabled = true
		c.ThinkingBudget = &budget
	}
}

// NewAnthropic creates a ChatProvider using Anthropic Messages API.
func NewAnthropic(model string, opts ...AnthropicOption) step.ChatProvider {
	cfg := AnthropicConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &anthropicProvider{model: model, cfg: cfg}
}

type anthropicProvider struct {
	model string
	cfg   AnthropicConfig
}

func (p *anthropicProvider) GenerateStream(ctx context.Context, req step.GenerateRequest) (step.AssistantStream, error) {
	// TODO: implement
	panic("not implemented")
}
