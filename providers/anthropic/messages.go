package anthropic

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
)

// Config configures Anthropic Messages API provider.
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

// WithThinking enables extended thinking.
func WithThinking(budget int) Option {
	return func(c *Config) {
		c.ThinkingEnabled = true
		c.ThinkingBudget = &budget
	}
}

// New creates a ChatProvider using Anthropic Messages API.
// It reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from environment if not explicitly set.
func New(model string, opts ...Option) step.ChatProvider {
	cfg := Config{}
	for _, opt := range opts {
		opt(&cfg)
	}

	// SDK auto-reads env vars; only override if explicitly set
	var clientOpts []option.RequestOption
	if cfg.APIKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.BaseURL))
	}
	client := anthropic.NewClient(clientOpts...)
	return &provider{model: model, cfg: cfg, client: client}
}

type provider struct {
	model  string
	cfg    Config
	client anthropic.Client
}

func (p *provider) GenerateStream(ctx context.Context, req step.GenerateRequest) (step.AssistantStream, error) {
	// TODO: implement
	panic("not implemented")
}
