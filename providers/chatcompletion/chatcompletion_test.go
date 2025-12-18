package chatcompletion_test

import (
	"testing"

	"github.com/inspirepan/step/internal/testutil"
	cc "github.com/inspirepan/step/providers/chatcompletion"
)

const envKey = "OPENAI_API_KEY"

func TestOpenAI_BasicTextGeneration(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := cc.New("gpt-4o-mini")
	cfg := testutil.DefaultConfig(provider)
	testutil.TestBasicTextGeneration(t, cfg)
}

func TestOpenAI_ToolCalling(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := cc.New("gpt-4o-mini")
	cfg := testutil.DefaultConfig(provider)
	testutil.TestToolCalling(t, cfg)
}

func TestOpenAI_SystemPrompt(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := cc.New("gpt-4o-mini")
	cfg := testutil.DefaultConfig(provider)
	testutil.TestSystemPrompt(t, cfg)
}

func TestOpenAI_MultiTurn(t *testing.T) {
	testutil.SkipIfNoEnv(t, envKey)

	provider := cc.New("gpt-4o-mini")
	cfg := testutil.DefaultConfig(provider)
	testutil.TestMultiTurn(t, cfg)
}
