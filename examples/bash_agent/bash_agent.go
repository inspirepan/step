package main

// Bash Agent Demo - A simple ReAct agent that executes bash commands.
// Requires OPENROUTER_API_KEY environment variable to be set.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/openrouter"
)

var (
	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("6")). // cyan
			Bold(true)

	assistantStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("5")) // magenta

	thinkingStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("8")). // bright black (gray)
			Italic(true)

	toolCallStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("3")). // yellow
			Bold(true)

	toolOutputStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("4")) // blue

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("1")). // red
			Bold(true)

	successStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("2")). // green
			Bold(true)
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
	userPrompt := "Please demonstrate a few harmless bash commands, such as checking the current directory, listing files, and showing the current date."
	fmt.Println(userStyle.Render("User: ") + userPrompt + "\n")
	ctx := context.Background()
	provider := openrouter.New("google/gemini-3-flash-preview", openrouter.WithReasoningEffort(openrouter.ReasoningEffortHigh))
	history := []step.Message{
		step.UserMessage{Parts: []step.Part{step.TextPart{Text: userPrompt}}},
	}
	onDelta := func(delta step.MessageDelta) {
		switch d := delta.(type) {
		case step.TextDelta:
			fmt.Print(assistantStyle.Render(d.Delta))
		case step.ThinkingDelta:
			fmt.Print(thinkingStyle.Render("\n"+strings.ReplaceAll(strings.TrimSpace(d.Delta), "\n\n", " ")) + "\n")
		case step.ToolCallDelta:
			if d.Name != "" {
				fmt.Printf("\n%s %s %s\n", toolCallStyle.Render("Tool:"), d.Name, d.ArgsDelta)
			}
		case step.ToolExecStartDelta:
		}
	}
	onMessage := func(msg step.Message) {
		if m, ok := msg.(step.ToolResultMessage); ok {
			if output := getToolResultText(m); output != "" {
				fmt.Println(toolOutputStyle.Render(output))
			}
		}
	}
	for {
		result, err := step.Step(ctx, step.StepRequest{
			Provider:     provider,
			SystemPrompt: "You're an agent running in user's shell with a bash tool. Be helpful and demonstrate commands step by step.",
			History:      history,
			Tools:        []step.Tool{&bashTool{}},
		}, step.WithOnDelta(onDelta), step.WithOnMessage(onMessage))

		if err != nil {
			fmt.Println(errorStyle.Render("Error: " + err.Error()))
			os.Exit(1)
		}

		history = append(history, result...)

		if !result.HasToolCall() {
			break
		}
	}
	fmt.Println(successStyle.Render("\nAgent completed."))
}

func getToolResultText(msg step.ToolResultMessage) string {
	for _, part := range msg.Parts {
		if textPart, ok := part.(step.TextPart); ok {
			return textPart.Text
		}
	}
	return ""
}
