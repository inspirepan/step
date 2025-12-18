package step

import (
	"context"
	"errors"
	"io"
)

// StepRequest configures a single agent step.
type StepRequest struct {
	Provider     ChatProvider
	SystemPrompt string
	History      []Message
	Tools        []Tool
}

// StepResult contains assistant output, tool calls and tool results.
type StepResult struct {
	Assistant   GenerateResult
	ToolCalls   []ToolCall
	ToolResults []ToolResult
	NewMessages []Message
	Cancelled   bool
}

// Step runs one step synchronously.
func Step(ctx context.Context, req StepRequest) (StepResult, error) {
	stream, err := StepStreamed(ctx, req)
	if err != nil {
		return StepResult{}, err
	}
	defer stream.Close()

	for {
		_, err := stream.Next(ctx)
		if err == nil {
			continue
		}
		if errors.Is(err, io.EOF) {
			break
		}
		return StepResult{}, err
	}

	res, err := stream.Result()
	if err != nil {
		return StepResult{}, err
	}
	return *res, nil
}
