package chatcompletion

import (
	"github.com/inspirepan/step"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// BuildMessages converts step request to OpenAI chat completion params.
func BuildMessages(
	req step.GenerateRequest,
	reasoningHandler ReasoningHandler,
	targetModel string,
	useCacheControl bool,
) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{}

	// System message
	if req.SystemPrompt != "" {
		if useCacheControl {
			// Use content array format with cache_control for OpenRouter
			textPart := openai.ChatCompletionContentPartTextParam{
				Text: req.SystemPrompt,
			}
			textPart.SetExtraFields(map[string]any{
				"cache_control": map[string]any{"type": "ephemeral"},
			})
			params.Messages = append(params.Messages, openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{textPart}))
		} else {
			params.Messages = append(params.Messages, openai.SystemMessage(req.SystemPrompt))
		}
	}

	// Convert history messages
	for _, msg := range req.History {
		switch m := msg.(type) {
		case step.UserMessage:
			params.Messages = append(params.Messages, convertUserMessage(m))
		case *step.UserMessage:
			params.Messages = append(params.Messages, convertUserMessage(*m))
		case step.AssistantMessage:
			params.Messages = append(params.Messages, convertAssistantMessage(m, reasoningHandler, targetModel))
		case *step.AssistantMessage:
			params.Messages = append(params.Messages, convertAssistantMessage(*m, reasoningHandler, targetModel))
		case step.ToolMessage:
			params.Messages = append(params.Messages, convertToolMessage(m))
		case *step.ToolMessage:
			params.Messages = append(params.Messages, convertToolMessage(*m))
		}
	}

	// Convert tools
	for _, tool := range req.Tools {
		params.Tools = append(params.Tools, convertToolSpec(tool))
	}

	if len(params.Tools) > 0 {
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		}
		params.ParallelToolCalls = openai.Bool(true)
	}

	// Add cache_control to the last user/tool message
	if useCacheControl {
		addCacheControlToLastMessage(params.Messages)
	}

	return params
}

// addCacheControlToLastMessage adds cache_control to the last text part of the last user/tool message.
func addCacheControlToLastMessage(messages []openai.ChatCompletionMessageParamUnion) {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := &messages[i]
		if msg.OfUser != nil {
			// User message: find and modify last text part
			if parts := msg.OfUser.Content.OfArrayOfContentParts; len(parts) > 0 {
				for j := len(parts) - 1; j >= 0; j-- {
					if parts[j].OfText != nil {
						parts[j].OfText.SetExtraFields(map[string]any{
							"cache_control": map[string]any{"type": "ephemeral"},
						})
						return
					}
				}
			}
			return
		}
		if msg.OfTool != nil {
			// Tool message: modify last part
			if parts := msg.OfTool.Content.OfArrayOfContentParts; len(parts) > 0 {
				parts[len(parts)-1].SetExtraFields(map[string]any{
					"cache_control": map[string]any{"type": "ephemeral"},
				})
			}
			return
		}
	}
}

func convertUserMessage(m step.UserMessage) openai.ChatCompletionMessageParamUnion {
	var parts []openai.ChatCompletionContentPartUnionParam

	for _, part := range m.Parts {
		switch p := part.(type) {
		case step.TextPart:
			parts = append(parts, openai.TextContentPart(p.Text))
		case *step.TextPart:
			parts = append(parts, openai.TextContentPart(p.Text))
		case step.ImagePart:
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: formatDataURL(p.MimeType, p.DataB64),
			}))
		case *step.ImagePart:
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: formatDataURL(p.MimeType, p.DataB64),
			}))
		}
	}

	if len(parts) == 0 {
		parts = append(parts, openai.TextContentPart(""))
	}

	return openai.UserMessage(parts)
}

func convertAssistantMessage(m step.AssistantMessage, handler ReasoningHandler, targetModel string) openai.ChatCompletionMessageParamUnion {
	msg := openai.ChatCompletionAssistantMessageParam{
		Role: "assistant",
	}

	var textContent string
	var thinkingParts []step.ThinkingPart
	var toolCalls []openai.ChatCompletionMessageToolCallUnionParam

	// Collect all parts
	for _, part := range m.Parts {
		switch p := part.(type) {
		case step.TextPart:
			textContent += p.Text
		case *step.TextPart:
			textContent += p.Text
		case step.ThinkingPart:
			thinkingParts = append(thinkingParts, p)
		case *step.ThinkingPart:
			thinkingParts = append(thinkingParts, *p)
		case step.ToolCallPart:
			toolCalls = append(toolCalls, convertToolCallPart(p))
		case *step.ToolCallPart:
			toolCalls = append(toolCalls, convertToolCallPart(*p))
		}
	}

	// Convert all thinking parts at once
	var degradedThinking string
	if handler != nil && len(thinkingParts) > 0 {
		key, value, degraded := handler.ConvertThinkingToExtra(thinkingParts, targetModel)
		degradedThinking = degraded

		// Add provider-specific extra fields (e.g. reasoning_details for OpenRouter)
		if key != "" && value != nil {
			msg.SetExtraFields(map[string]any{
				key: value,
			})
		}
	}

	// Build content: prepend degraded thinking if any
	fullContent := degradedThinking + textContent
	if fullContent != "" {
		msg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String(fullContent),
		}
	}

	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return openai.ChatCompletionMessageParamUnion{OfAssistant: &msg}
}

func convertToolCallPart(p step.ToolCallPart) openai.ChatCompletionMessageToolCallUnionParam {
	return openai.ChatCompletionMessageToolCallUnionParam{
		OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
			ID: p.CallID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      p.Name,
				Arguments: string(p.ArgsJSON),
			},
		},
	}
}

func convertToolMessage(m step.ToolMessage) openai.ChatCompletionMessageParamUnion {
	var content string
	for _, part := range m.Parts {
		switch p := part.(type) {
		case step.TextPart:
			content += p.Text
		case *step.TextPart:
			content += p.Text
		}
	}
	if content == "" {
		content = "<system-reminder>Tool ran without output or errors</system-reminder>"
	}
	return openai.ToolMessage(content, m.CallID)
}

func convertToolSpec(spec step.ToolSpec) openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        spec.Name,
		Description: openai.String(spec.Description),
		Parameters:  shared.FunctionParameters(spec.Parameters),
	})
}

func formatDataURL(mimeType, dataB64 string) string {
	return "data:" + mimeType + ";base64," + dataB64
}
