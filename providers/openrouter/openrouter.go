package openrouter

import (
	"context"
	"os"
	"strings"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	cc "github.com/inspirepan/step/providers/chatcompletion"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func isClaudeModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "claude")
}

func isGeminiModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "gemini")
}

const defaultBaseURL = "https://openrouter.ai/api/v1"

// ReasoningEffort defines the effort level for reasoning models.
// Supported by GPT-5 series and Gemini 3 series.
type ReasoningEffort string

const (
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
)

// Verbosity defines the verbosity level for token efficiency control.
// Supported by GPT-5 and Claude Opus 4.5 (mapped from Effort parameter).
type Verbosity string

const (
	VerbosityHigh   Verbosity = "high"
	VerbosityMedium Verbosity = "medium"
	VerbosityLow    Verbosity = "low"
)

// ProviderSortStrategy defines the sorting strategy for provider routing.
type ProviderSortStrategy string

const (
	ProviderSortPrice      ProviderSortStrategy = "price"
	ProviderSortThroughput ProviderSortStrategy = "throughput"
	ProviderSortLatency    ProviderSortStrategy = "latency"
)

// AnthropicThinkingConfig configures thinking/reasoning for Anthropic models.
type AnthropicThinkingConfig struct {
	Enable    bool
	MaxTokens int
}

// ProviderRouting configures OpenRouter's provider routing preferences.
type ProviderRouting struct {
	Order  []string             // Preferred provider order
	Only   []string             // Only use these providers
	Ignore []string             // Ignore these providers
	Sort   ProviderSortStrategy // Sorting strategy when order is not specified
}

// Config configures OpenRouter API provider.
type Config struct {
	base.Config

	// OpenRouter-specific options
	AnthropicThinking *AnthropicThinkingConfig
	ReasoningEffort   ReasoningEffort
	Verbosity         Verbosity
	ProviderRouting   *ProviderRouting
}

// Option is a functional option for this provider.
type Option func(*Config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Config) { c.APIKey = key }
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

// WithThinkingBudget configures thinking/reasoning budget for Anthropic models.
// See: https://openrouter.ai/docs/use-cases/reasoning-tokens#anthropic-models-with-reasoning-tokens
func WithThinkingBudget(maxTokens int) Option {
	return func(c *Config) {
		c.AnthropicThinking = &AnthropicThinkingConfig{
			Enable:    true,
			MaxTokens: maxTokens,
		}
	}
}

// WithReasoningEffort sets the reasoning effort level for reasoning models.
// Supported by GPT-5 series and Gemini 3 series.
func WithReasoningEffort(effort ReasoningEffort) Option {
	return func(c *Config) {
		c.ReasoningEffort = effort
	}
}

// WithVerbosity sets the verbosity level for token efficiency control.
// Supported by GPT-5 and Claude Opus 4.5 (mapped from Effort parameter).
func WithVerbosity(verbosity Verbosity) Option {
	return func(c *Config) {
		c.Verbosity = verbosity
	}
}

// WithProviderSorting sets the provider sorting strategy.
// Valid values: "price", "throughput", "latency".
func WithProviderSorting(strategy ProviderSortStrategy) Option {
	return func(c *Config) {
		if c.ProviderRouting == nil {
			c.ProviderRouting = &ProviderRouting{}
		}
		c.ProviderRouting.Sort = strategy
	}
}

// WithProviderOnly restricts to only use the specified providers.
func WithProviderOnly(providers ...string) Option {
	return func(c *Config) {
		if c.ProviderRouting == nil {
			c.ProviderRouting = &ProviderRouting{}
		}
		c.ProviderRouting.Only = providers
	}
}

// WithProviderOrder sets the preferred provider order.
func WithProviderOrder(providers ...string) Option {
	return func(c *Config) {
		if c.ProviderRouting == nil {
			c.ProviderRouting = &ProviderRouting{}
		}
		c.ProviderRouting.Order = providers
	}
}

// WithProviderIgnore sets providers to ignore.
func WithProviderIgnore(providers ...string) Option {
	return func(c *Config) {
		if c.ProviderRouting == nil {
			c.ProviderRouting = &ProviderRouting{}
		}
		c.ProviderRouting.Ignore = providers
	}
}

// New creates a Provider using OpenRouter API.
// It reads OPENROUTER_API_KEY from environment if not explicitly set.
// BaseURL is fixed to https://openrouter.ai/api/v1.
func New(model string, opts ...Option) step.Provider {
	cfg := Config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}

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

	// Add Anthropic beta headers for Claude models
	if isClaudeModel(model) {
		clientOpts = append(clientOpts, option.WithHeader(
			"x-anthropic-beta",
			"fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
		))
	}

	// Apply OpenRouter-specific options to ExtraBody
	// Include usage info to get cache tokens at the end of the response
	clientOpts = append(clientOpts, option.WithJSONSet("usage", map[string]any{
		"include": true,
	}))

	if cfg.AnthropicThinking != nil {
		clientOpts = append(clientOpts, option.WithJSONSet("reasoning", map[string]any{
			"enable":     cfg.AnthropicThinking.Enable,
			"max_tokens": cfg.AnthropicThinking.MaxTokens,
		}))
	} else if cfg.ReasoningEffort != "" {
		clientOpts = append(clientOpts, option.WithJSONSet("reasoning", map[string]any{
			"effort": string(cfg.ReasoningEffort),
		}))
	}

	if cfg.Verbosity != "" {
		clientOpts = append(clientOpts, option.WithJSONSet("verbosity", string(cfg.Verbosity)))
	}

	if cfg.ProviderRouting != nil {
		provider := make(map[string]any)
		if len(cfg.ProviderRouting.Order) > 0 {
			provider["order"] = cfg.ProviderRouting.Order
		}
		if len(cfg.ProviderRouting.Only) > 0 {
			provider["only"] = cfg.ProviderRouting.Only
		}
		if len(cfg.ProviderRouting.Ignore) > 0 {
			provider["ignore"] = cfg.ProviderRouting.Ignore
		}
		if cfg.ProviderRouting.Sort != "" {
			provider["sort"] = string(cfg.ProviderRouting.Sort)
		}
		if len(provider) > 0 {
			clientOpts = append(clientOpts, option.WithJSONSet("provider", provider))
		}
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
	handler := NewReasoningHandler(p.model)
	// Enable cache_control for Claude and Gemini models via OpenRouter
	useCacheControl := isClaudeModel(p.model) || isGeminiModel(p.model)
	params := cc.BuildMessages(req, handler, p.model, useCacheControl)
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
		rec.Provider = "openrouter"
		rec.Model = p.model
		_ = debug.Log(rec)
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	return cc.NewStream("openrouter", p.model, stream, handler, debug), nil
}
