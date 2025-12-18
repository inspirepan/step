package base

import (
	"os"

	"github.com/joho/godotenv"
)

func init() {
	// Auto-load .env file if it exists (silent fail)
	_ = godotenv.Load()
}

// LoadEnv loads environment variables from specified .env files.
// If no files are specified, it loads from .env in the current directory.
func LoadEnv(filenames ...string) error {
	return godotenv.Load(filenames...)
}

// Config contains common configuration for all providers.
type Config struct {
	APIKey  string
	BaseURL string

	// Debug options
	// DebugPath writes JSONL debug records (request/chunk/event) when set.
	DebugPath string

	// Generation options
	MaxOutputTokens *int
	Temperature     *float64

	// Extra options
	ExtraHeaders map[string]string
	ExtraBody    map[string]any
}

// ApplyEnvDefaults applies environment variable defaults if config values are empty.
func ApplyEnvDefaults(cfg *Config, apiKeyEnv, baseURLEnv string) {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv(apiKeyEnv)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = os.Getenv(baseURLEnv)
	}
}
