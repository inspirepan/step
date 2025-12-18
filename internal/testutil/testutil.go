// Package testutil provides common testing utilities for provider tests.
package testutil

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/inspirepan/step"
)

const DefaultTimeout = 60 * time.Second

// SkipIfNoEnv skips the test if the environment variable is not set.
func SkipIfNoEnv(t *testing.T, envVar string) {
	t.Helper()
	if os.Getenv(envVar) == "" {
		t.Skipf("skipping: %s not set", envVar)
	}
}

// TestConfig holds configuration for a test run.
type TestConfig struct {
	Provider step.ChatProvider
	Timeout  time.Duration
}

// DefaultConfig returns a TestConfig with default timeout.
func DefaultConfig(provider step.ChatProvider) TestConfig {
	return TestConfig{
		Provider: provider,
		Timeout:  DefaultTimeout,
	}
}

// TestBasicTextGeneration tests basic text generation capability.
func TestBasicTextGeneration(t *testing.T, cfg TestConfig) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	req := step.GenerateRequest{
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "Write a haiku"}}},
		},
	}

	stream, err := cfg.Provider.GenerateStream(ctx, req)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	defer stream.Close()

	var textContent strings.Builder
	var eventCount int
	var sawTextStart bool
	var sawTextEnd bool

	for {
		ev, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("stream.Next failed: %v", err)
		}
		eventCount++
		switch ev.Type {
		case step.EventTextStart:
			sawTextStart = true
		case step.EventTextEnd:
			sawTextEnd = true
		case step.EventTextDelta:
			textContent.WriteString(ev.Delta)
		}
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("stream.Result failed: %v", err)
	}

	if result.Usage == nil {
		t.Log("warning: usage info not returned")
	} else {
		if result.Usage.OutputTokens == 0 {
			t.Error("expected non-zero output tokens")
		}
	}

	text := textContent.String()
	if text == "" {
		t.Error("expected non-empty text response")
	}
	if !sawTextStart {
		t.Error("expected text_start event")
	}
	if !sawTextEnd {
		t.Error("expected text_end event")
	}

	t.Logf("response: %q (events: %d)", text, eventCount)
}

// calculatorTool is a simple test tool for tool calling tests.
type calculatorTool struct{}

func (c calculatorTool) Spec() step.ToolSpec {
	return step.ToolSpec{
		Name:        "add",
		Description: "Add two numbers together",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number", "description": "First number"},
				"b": map[string]any{"type": "number", "description": "Second number"},
			},
			"required": []string{"a", "b"},
		},
	}
}

func (c calculatorTool) Execute(_ context.Context, call step.ToolCall) (step.ToolResult, error) {
	var args struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}
	if err := json.Unmarshal(call.ArgsJSON, &args); err != nil {
		return step.ToolResult{CallID: call.CallID, IsError: true}, err
	}
	result := args.A + args.B
	return step.ToolResult{
		CallID: call.CallID,
		Name:   call.Name,
		Parts:  []step.Part{step.TextPart{Text: fmt.Sprintf("%.2f", result)}},
	}, nil
}

// TestToolCalling tests tool calling capability.
func TestToolCalling(t *testing.T, cfg TestConfig) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	tool := calculatorTool{}
	req := step.GenerateRequest{
		SystemPrompt: "You are a helpful assistant. Use the add tool when asked to add numbers.",
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "What is 123 + 456 and 444+888, use calculator pls?"}}},
		},
		Tools: []step.ToolSpec{tool.Spec()},
	}

	stream, err := cfg.Provider.GenerateStream(ctx, req)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	defer stream.Close()

	var toolCalls []step.ToolCallPart
	var sawToolCallStart bool

	for {
		ev, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("stream.Next failed: %v", err)
		}

		switch ev.Type {
		case step.EventToolCallStart:
			sawToolCallStart = true
		case step.EventToolCallEnd:
			if ev.ToolCall != nil {
				toolCalls = append(toolCalls, *ev.ToolCall)
			}
		}
	}

	result, err := stream.Result()
	if err != nil {
		t.Fatalf("stream.Result failed: %v", err)
	}

	if result.StopReason != step.StopToolUse {
		t.Errorf("expected StopToolUse, got %s", result.StopReason)
	}

	if !sawToolCallStart {
		t.Error("expected toolcall_start event")
	}

	if len(toolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}

	// Verify tool call content
	tc := toolCalls[0]
	if tc.Name != "add" {
		t.Errorf("expected tool name 'add', got %q", tc.Name)
	}

	t.Logf("tool calls: %d, first call: %s(%s)", len(toolCalls), tc.Name, string(tc.ArgsJSON))
}

// TestSystemPrompt tests that system prompt is respected.
func TestSystemPrompt(t *testing.T, cfg TestConfig) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	req := step.GenerateRequest{
		SystemPrompt: "You are a pirate. Always respond like a pirate. Use 'Arrr' in your response.",
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "Hello, how are you?"}}},
		},
	}

	stream, err := cfg.Provider.GenerateStream(ctx, req)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	defer stream.Close()

	var response strings.Builder
	for {
		ev, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("stream.Next failed: %v", err)
		}

		if ev.Type == step.EventTextDelta {
			response.WriteString(ev.Delta)
		}
	}

	text := strings.ToLower(response.String())
	if !strings.Contains(text, "arrr") && !strings.Contains(text, "ahoy") && !strings.Contains(text, "matey") {
		t.Errorf("expected pirate-like response, got: %s", response.String())
	}

	t.Logf("response: %s", response.String())
}

// TestMultiTurn tests multi-turn conversation.
func TestMultiTurn(t *testing.T, cfg TestConfig) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	req := step.GenerateRequest{
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "My name is Alice."}}},
			step.AssistantMessage{Parts: []step.Part{step.TextPart{Text: "Hello Alice! Nice to meet you."}}},
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "What is my name?"}}},
		},
	}

	stream, err := cfg.Provider.GenerateStream(ctx, req)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	defer stream.Close()

	var response strings.Builder
	for {
		ev, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("stream.Next failed: %v", err)
		}

		if ev.Type == step.EventTextDelta {
			response.WriteString(ev.Delta)
		}
	}

	text := response.String()
	if !strings.Contains(strings.ToLower(text), "alice") {
		t.Errorf("expected response to contain 'Alice', got: %s", text)
	}

	t.Logf("response: %s", text)
}
