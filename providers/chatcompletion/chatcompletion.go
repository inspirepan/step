package chatcompletion

import (
	"context"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Config configures OpenAI Chat Completions API provider.
type Config struct {
	base.Config
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

// New creates a Provider using OpenAI Chat Completions API.
// It reads OPENAI_API_KEY and OPENAI_BASE_URL from environment if not explicitly set.
func New(model string, opts ...Option) step.Provider {
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

func (p *provider) Stream(ctx context.Context, req step.ProviderRequest) (step.ProviderStream, error) {
	reasoningHandler := NewDefaultReasoningHandler(p.model)
	params := BuildMessages(req, reasoningHandler, p.model, false)
	params.Model = p.model

	// Apply config options
	if p.cfg.Temperature != nil {
		params.Temperature = openai.Float(*p.cfg.Temperature)
	}
	if p.cfg.MaxOutputTokens != nil {
		params.MaxTokens = openai.Int(int64(*p.cfg.MaxOutputTokens))
	}

	debug, err := base.NewDebugLogger(p.cfg.DebugPath)
	if err != nil {
		return nil, err
	}
	if debug != nil {
		rec := base.NewDebugRecord("request", params)
		rec.Provider = "chatcompletion"
		rec.Model = p.model
		_ = debug.Log(rec)
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	return NewStream("chatcompletion", p.model, stream, reasoningHandler, debug), nil
}
