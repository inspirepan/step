package providers

import (
	"context"

	"github.com/inspirepan/step"
)

// GoogleConfig configures Google Gemini API provider.
type GoogleConfig struct {
	APIKey  string
	BaseURL string

	// Generation options
	MaxOutputTokens *int
	Temperature     *float64

	// Thinking options
	ThinkingEnabled bool
	ThinkingBudget  *int
}

// GoogleOption is a functional option for Google provider.
type GoogleOption func(*GoogleConfig)

// GoogleWithAPIKey sets the API key.
func GoogleWithAPIKey(key string) GoogleOption {
	return func(c *GoogleConfig) { c.APIKey = key }
}

// GoogleWithBaseURL sets a custom base URL.
func GoogleWithBaseURL(url string) GoogleOption {
	return func(c *GoogleConfig) { c.BaseURL = url }
}

// GoogleWithTemperature sets the temperature.
func GoogleWithTemperature(t float64) GoogleOption {
	return func(c *GoogleConfig) { c.Temperature = &t }
}

// GoogleWithMaxOutputTokens sets the max output tokens.
func GoogleWithMaxOutputTokens(n int) GoogleOption {
	return func(c *GoogleConfig) { c.MaxOutputTokens = &n }
}

// GoogleWithThinking enables thinking mode.
func GoogleWithThinking(budget int) GoogleOption {
	return func(c *GoogleConfig) {
		c.ThinkingEnabled = true
		c.ThinkingBudget = &budget
	}
}

// NewGoogle creates a ChatProvider using Google Gemini API.
func NewGoogle(model string, opts ...GoogleOption) step.ChatProvider {
	cfg := GoogleConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &googleProvider{model: model, cfg: cfg}
}

type googleProvider struct {
	model string
	cfg   GoogleConfig
}

func (p *googleProvider) GenerateStream(ctx context.Context, req step.GenerateRequest) (step.AssistantStream, error) {
	// TODO: implement
	panic("not implemented")
}
