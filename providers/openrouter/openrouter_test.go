package openrouter_test

import (
	"testing"

	"github.com/inspirepan/step/internal/testutil"
	"github.com/inspirepan/step/providers/openrouter"
)

const envKey = "OPENROUTER_API_KEY"

func TestOpenRouter_BasicTextGeneration(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New(
		"google/gemini-3-flash-preview",
		openrouter.WithReasoningEffort(openrouter.ReasoningEffortMinimal),
		openrouter.WithDebug("openrouter.debug.log"),
	)
	cfg := testutil.DefaultConfig(provider)
	testutil.TestBasicTextGeneration(t, cfg)
}

func TestOpenRouter_ToolCalling(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New("google/gemini-3-flash-preview", openrouter.WithReasoningEffort(openrouter.ReasoningEffortMinimal))
	cfg := testutil.DefaultConfig(provider)
	testutil.TestToolCalling(t, cfg)
}

func TestOpenRouter_SystemPrompt(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New("google/gemini-3-flash-preview", openrouter.WithReasoningEffort(openrouter.ReasoningEffortMinimal))
	cfg := testutil.DefaultConfig(provider)
	testutil.TestSystemPrompt(t, cfg)
}

func TestOpenRouter_MultiTurn(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New("google/gemini-3-flash-preview", openrouter.WithReasoningEffort(openrouter.ReasoningEffortMinimal))
	cfg := testutil.DefaultConfig(provider)
	testutil.TestMultiTurn(t, cfg)
}

// TestOpenRouter_Claude tests Claude models via OpenRouter.
func TestOpenRouter_Claude(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New("anthropic/claude-3.5-haiku")
	cfg := testutil.DefaultConfig(provider)
	testutil.TestBasicTextGeneration(t, cfg)
}

// TestOpenRouter_ClaudeWithThinking tests Claude models with thinking enabled.
func TestOpenRouter_ClaudeWithThinking(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := openrouter.New(
		"anthropic/claude-sonnet-4",
		openrouter.WithThinkingBudget(5000),
	)
	cfg := testutil.DefaultConfig(provider)
	testutil.TestBasicTextGeneration(t, cfg)
}
