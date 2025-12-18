package responses

import (
	"context"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// Config configures OpenAI Responses API provider.
type Config struct {
	base.Config

	// Reasoning options
	Reasoning shared.ReasoningParam
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

// WithReasoningEffort sets reasoning effort level.
func WithReasoningEffort(effort shared.ReasoningEffort) Option {
	return func(c *Config) { c.Reasoning.Effort = effort }
}

// WithReasoningSummary sets reasoning summary mode.
func WithReasoningSummary(summary shared.ReasoningSummary) Option {
	return func(c *Config) { c.Reasoning.Summary = summary }
}

// New creates a ChatProvider using OpenAI Responses API.
// It reads OPENAI_API_KEY and OPENAI_BASE_URL from environment if not explicitly set.
func New(model string, opts ...Option) step.ChatProvider {
	cfg := Config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	base.ApplyEnvDefaults(&cfg.Config, "OPENAI_API_KEY", "OPENAI_BASE_URL")

	var clientOpts []option.RequestOption
	if cfg.APIKey != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.BaseURL))
	}
	for k, v := range cfg.ExtraHeaders {
		clientOpts = append(clientOpts, option.WithHeader(k, v))
	}
	for k, v := range cfg.ExtraBody {
		clientOpts = append(clientOpts, option.WithJSONSet(k, v))
	}
	client := openai.NewClient(clientOpts...)
	return &provider{model: model, cfg: cfg, client: client}
}

type provider struct {
	model  string
	cfg    Config
	client openai.Client
}

func (p *provider) GenerateStream(ctx context.Context, req step.GenerateRequest) (step.AssistantStream, error) {
	// TODO: implement
	panic("not implemented")
}
