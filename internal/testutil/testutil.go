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
	Provider step.Provider
	Timeout  time.Duration
}

// DefaultConfig returns a TestConfig with default timeout.
func DefaultConfig(provider step.Provider) TestConfig {
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

	req := step.ProviderRequest{
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "Write a haiku"}}},
		},
	}

	stream, err := cfg.Provider.Stream(ctx, req)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var textContent strings.Builder
	var assistantMsg *step.AssistantMessage

	for {
		up, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				if up == nil {
					break
				}
			} else {
				t.Fatalf("stream.Next failed: %v", err)
			}
		}
		if up != nil {
			switch u := up.(type) {
			case step.ProviderDeltaUpdate:
				switch d := u.Delta.(type) {
				case step.TextDelta:
					textContent.WriteString(d.Delta)
				}
			case step.ProviderMessageUpdate:
				msg := u.Message
				assistantMsg = &msg
			}
		}
		if err != nil && errors.Is(err, io.EOF) {
			break
		}
	}

	text := textContent.String()
	if text == "" {
		// Some providers may not stream text deltas; fall back to the final message.
		if assistantMsg == nil {
			t.Fatal("expected assistant message")
		}
		for _, part := range assistantMsg.Parts {
			p, ok := part.(step.TextPart)
			if ok {
				text += p.Text
			}
			pp, ok := part.(*step.TextPart)
			if ok && pp != nil {
				text += pp.Text
			}
		}
		if text == "" {
			t.Error("expected non-empty text response")
		}
	}

	if assistantMsg != nil {
		if assistantMsg.Usage == nil {
			t.Log("warning: usage info not returned")
		} else if assistantMsg.Usage.OutputTokens == 0 {
			t.Error("expected non-zero output tokens")
		}
	}

	t.Logf("response: %q", text)
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
	req := step.ProviderRequest{
		SystemPrompt: "You are a helpful assistant. Use the add tool when asked to add numbers.",
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "What is 123 + 456 and 444+888, use calculator pls?"}}},
		},
		Tools: []step.ToolSpec{tool.Spec()},
	}

	stream, err := cfg.Provider.Stream(ctx, req)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var assistantMsg *step.AssistantMessage
	var sawToolCallDelta bool

	for {
		up, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				if up == nil {
					break
				}
			} else {
				t.Fatalf("stream.Next failed: %v", err)
			}
		}
		if up != nil {
			switch u := up.(type) {
			case step.ProviderDeltaUpdate:
				switch u.Delta.(type) {
				case step.ToolCallDelta:
					sawToolCallDelta = true
				}
			case step.ProviderMessageUpdate:
				msg := u.Message
				assistantMsg = &msg
			}
		}
		if err != nil && errors.Is(err, io.EOF) {
			break
		}
	}
	if assistantMsg == nil {
		t.Fatal("expected assistant message")
	}

	var toolCalls []step.ToolCallPart
	for _, part := range assistantMsg.Parts {
		tc, ok := part.(step.ToolCallPart)
		if ok {
			toolCalls = append(toolCalls, tc)
			continue
		}
		ptc, ok := part.(*step.ToolCallPart)
		if ok && ptc != nil {
			toolCalls = append(toolCalls, *ptc)
		}
	}

	if !sawToolCallDelta && len(toolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	if len(toolCalls) > 0 {
		if toolCalls[0].Name != "add" {
			t.Errorf("expected tool name 'add', got %q", toolCalls[0].Name)
		}
		t.Logf("tool calls: %d, first call: %s(%s)", len(toolCalls), toolCalls[0].Name, string(toolCalls[0].ArgsJSON))
	}
}

// TestSystemPrompt tests that system prompt is respected.
func TestSystemPrompt(t *testing.T, cfg TestConfig) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	req := step.ProviderRequest{
		SystemPrompt: "You are a pirate. Always respond like a pirate. Use 'Arrr' in your response.",
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "Hello, how are you?"}}},
		},
	}

	stream, err := cfg.Provider.Stream(ctx, req)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var response strings.Builder
	for {
		up, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				if up == nil {
					break
				}
			} else {
				t.Fatalf("stream.Next failed: %v", err)
			}
		}
		if up != nil {
			if dUp, ok := up.(step.ProviderDeltaUpdate); ok {
				if d, ok := dUp.Delta.(step.TextDelta); ok {
					response.WriteString(d.Delta)
				}
			}
		}
		if err != nil && errors.Is(err, io.EOF) {
			break
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

	req := step.ProviderRequest{
		History: []step.Message{
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "My name is Alice."}}},
			step.AssistantMessage{Parts: []step.Part{step.TextPart{Text: "Hello Alice! Nice to meet you."}}},
			step.UserMessage{Parts: []step.Part{step.TextPart{Text: "What is my name?"}}},
		},
	}

	stream, err := cfg.Provider.Stream(ctx, req)
	if err != nil {
		t.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	var response strings.Builder
	for {
		up, err := stream.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				if up == nil {
					break
				}
			} else {
				t.Fatalf("stream.Next failed: %v", err)
			}
		}
		if up != nil {
			if dUp, ok := up.(step.ProviderDeltaUpdate); ok {
				if d, ok := dUp.Delta.(step.TextDelta); ok {
					response.WriteString(d.Delta)
				}
			}
		}
		if err != nil && errors.Is(err, io.EOF) {
			break
		}
	}

	text := response.String()
	if !strings.Contains(strings.ToLower(text), "alice") {
		t.Errorf("expected response to contain 'Alice', got: %s", text)
	}

	t.Logf("response: %s", text)
}
