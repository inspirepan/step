// Bash Agent Demo - A simple ReAct agent that executes bash commands.
// Requires OPENROUTER_API_KEY environment variable to be set.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/openrouter"
)

type bashTool struct{}

var _ step.Tool = (*bashTool)(nil)

func (b *bashTool) Spec() step.ToolSpec {
	return step.ToolSpec{
		Name:        "Bash",
		Description: "Run a bash command",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The bash command to run",
				},
			},
			"required": []string{"command"},
		},
	}
}

type bashArgs struct {
	Command string `json:"command"`
}

func (b *bashTool) Execute(ctx context.Context, call step.ToolCallPart) (step.ToolResult, error) {
	var args bashArgs
	if err := json.Unmarshal(call.ArgsJSON, &args); err != nil {
		return step.ToolResult{
			CallID:  call.CallID,
			Name:    call.Name,
			IsError: true,
			Parts:   []step.Part{step.TextPart{Text: "failed to parse arguments: " + err.Error()}},
		}, nil
	}

	cmd := exec.CommandContext(ctx, "bash", "-c", args.Command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return step.ToolResult{
			CallID:  call.CallID,
			Name:    call.Name,
			IsError: true,
			Parts:   []step.Part{step.TextPart{Text: string(output) + "\n" + err.Error()}},
		}, nil
	}

	return step.ToolResult{
		CallID: call.CallID,
		Name:   call.Name,
		Parts:  []step.Part{step.TextPart{Text: string(output)}},
	}, nil
}



func main() {
	fmt.Fprintln(os.Stderr, "[DEBUG] main started")
	userPrompt := "Please demonstrate a few harmless bash commands, such as checking the current directory, listing files, and showing the current date."
	fmt.Printf("User: %s\n\n", userPrompt)

	ctx := context.Background()
	fmt.Fprintln(os.Stderr, "[DEBUG] Creating provider...")
	provider := openrouter.New("google/gemini-3-flash-preview", openrouter.WithReasoningEffort(openrouter.ReasoningEffortHigh))
	fmt.Println("[DEBUG] Provider created")

	tools := []step.Tool{&bashTool{}}
	history := []step.Message{
		step.UserMessage{Parts: []step.Part{step.TextPart{Text: userPrompt}}},
	}

	systemPrompt := "You're an agent running in user's shell with a bash tool. Be helpful and demonstrate commands step by step."

	for {
		fmt.Println("[DEBUG] Calling step.Step...")
		result, err := step.Step(ctx, step.StepRequest{
			Provider:     provider,
			SystemPrompt: systemPrompt,
			History:      history,
			Tools:        tools,
		})
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}

		// append step result to history
		history = append(history, result...)

		// print messages
		for _, msg := range result {
			switch m := msg.(type) {
			case step.AssistantMessage:
				for _, part := range m.Parts {
					switch p := part.(type) {
					case step.TextPart:
						if p.Text != "" {
							fmt.Printf("Assistant: %s\n", p.Text)
						}
					case step.ToolCallPart:
						fmt.Printf("Calling tool: %s\n", p.Name)
					}
				}
			case step.ToolResultMessage:
				fmt.Printf("Tool output:\n%s\n", getToolResultText(m))
			}
		}

		if !result.HasToolCall() {
			fmt.Println("Agent completed.")
			return
		}
	}
}

func getToolResultText(msg step.ToolResultMessage) string {
	for _, part := range msg.Parts {
		if textPart, ok := part.(step.TextPart); ok {
			return textPart.Text
		}
	}
	return ""
}
