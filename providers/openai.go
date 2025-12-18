package providers

import (
	"context"

	"github.com/inspirepan/step"
)

// OpenAIConfig configures OpenAI Responses API provider.
type OpenAIConfig struct {
	APIKey  string
	BaseURL string

	// Generation options
	MaxOutputTokens *int
	Temperature     *float64

	// Reasoning options
	ReasoningEffort string // "low", "medium", "high"
}

// OpenAIOption is a functional option for OpenAI provider.
type OpenAIOption func(*OpenAIConfig)

// OpenAIWithAPIKey sets the API key.
func OpenAIWithAPIKey(key string) OpenAIOption {
	return func(c *OpenAIConfig) { c.APIKey = key }
}

// OpenAIWithBaseURL sets a custom base URL.
func OpenAIWithBaseURL(url string) OpenAIOption {
	return func(c *OpenAIConfig) { c.BaseURL = url }
}

// OpenAIWithTemperature sets the temperature.
func OpenAIWithTemperature(t float64) OpenAIOption {
	return func(c *OpenAIConfig) { c.Temperature = &t }
}

// OpenAIWithMaxOutputTokens sets the max output tokens.
func OpenAIWithMaxOutputTokens(n int) OpenAIOption {
	return func(c *OpenAIConfig) { c.MaxOutputTokens = &n }
}

// NewOpenAI creates a ChatProvider using OpenAI Responses API.
func NewOpenAI(model string, opts ...OpenAIOption) step.ChatProvider {
	cfg := OpenAIConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &openaiProvider{model: model, cfg: cfg}
}

type openaiProvider struct {
	model string
	cfg   OpenAIConfig
}

func (p *openaiProvider) GenerateStream(ctx context.Context, req step.GenerateRequest) (step.AssistantStream, error) {
	// TODO: implement
	panic("not implemented")
}
