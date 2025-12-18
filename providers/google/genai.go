package google

import (
	"context"
	"errors"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
)

// Config configures Google Generative AI API provider.
type Config struct {
	base.Config

	// Thinking options
	ThinkingEnabled bool
	ThinkingBudget  *int
}

// Option is a functional option for this provider.
type Option func(*Config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Config) { c.APIKey = key }
}

// WithBaseURL sets a custom base URL.
func WithBaseURL(url string) Option {
	return func(c *Config) { c.BaseURL = url }
}

// WithTemperature sets the temperature.
func WithTemperature(t float64) Option {
	return func(c *Config) { c.Temperature = &t }
}

// WithMaxOutputTokens sets the max output tokens.
func WithMaxOutputTokens(n int) Option {
	return func(c *Config) { c.MaxOutputTokens = &n }
}

// WithDebug enables JSONL debug logging to the specified file path.
func WithDebug(path string) Option {
	return func(c *Config) { c.DebugPath = path }
}

// WithExtraHeader adds a custom header to requests.
func WithExtraHeader(key, value string) Option {
	return func(c *Config) {
		if c.ExtraHeaders == nil {
			c.ExtraHeaders = make(map[string]string)
		}
		c.ExtraHeaders[key] = value
	}
}

// WithExtraBody adds a custom field to the request body.
func WithExtraBody(key string, value any) Option {
	return func(c *Config) {
		if c.ExtraBody == nil {
			c.ExtraBody = make(map[string]any)
		}
		c.ExtraBody[key] = value
	}
}

// WithThinking enables thinking mode.
func WithThinking(budget int) Option {
	return func(c *Config) {
		c.ThinkingEnabled = true
		c.ThinkingBudget = &budget
	}
}

// New creates a Provider using Google Generative AI API.
// It reads GEMINI_API_KEY (or GOOGLE_API_KEY) and GEMINI_BASE_URL from environment if not explicitly set.
func New(model string, opts ...Option) step.Provider {
	cfg := Config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	base.ApplyEnvDefaults(&cfg.Config, "GEMINI_API_KEY", "GEMINI_BASE_URL")
	if cfg.APIKey == "" {
		base.ApplyEnvDefaults(&cfg.Config, "GOOGLE_API_KEY", "")
	}
	return &provider{model: model, cfg: cfg}
}

type provider struct {
	model string
	cfg   Config
}

func (p *provider) Stream(ctx context.Context, req step.ProviderRequest) (step.ProviderStream, error) {
	_ = ctx
	_ = req
	return nil, errors.New("step/providers/google: not implemented")
}
