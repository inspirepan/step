package chatcompletion

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

// stream stage
type streamStage int

const (
	stageWaiting streamStage = iota
	stageThinking
	stageText
	stageTool
)

// Stream implements step.AssistantStream for Chat Completion API.
type Stream struct {
	providerName     string
	modelName        string
	stream           *ssestream.Stream[openai.ChatCompletionChunk]
	reasoningHandler ReasoningHandler
	debug            *base.DebugLogger

	mu    sync.Mutex
	stage streamStage
	done  bool
	err           error
	pendingEvents []step.AssistantEvent

	// Accumulators
	textContent      []string
	toolCalls        map[int]*toolCallAccumulator
	emittedToolStart map[int]bool
	currentToolIdx   int  // current tool call index being processed
	hasCurrentTool   bool // whether currentToolIdx is valid

	// Final result
	stopReason step.StopReason
	usage      *step.Usage
	parts      []step.Part
}

type toolCallAccumulator struct {
	id      string
	name    string
	argsStr string
}

// NewStream creates a new Stream wrapper.
func NewStream(
	providerName string,
	modelName string,
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	handler ReasoningHandler,
	debug *base.DebugLogger,
) *Stream {
	if handler == nil {
		handler = &NoOpReasoningHandler{}
	}
	return &Stream{
		providerName:     providerName,
		modelName:        modelName,
		stream:           stream,
		reasoningHandler: handler,
		debug:            debug,
		stage:            stageWaiting,
		toolCalls:        make(map[int]*toolCallAccumulator),
		emittedToolStart: make(map[int]bool),
	}
}

// Next returns the next event from the stream.
func (s *Stream) Next(ctx context.Context) (step.AssistantEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Return pending events first (including after finalize)
	if len(s.pendingEvents) > 0 {
		return s.dequeuePendingEvent()
	}

	if s.done {
		return step.AssistantEvent{Type: step.EventDone, Reason: s.stopReason}, io.EOF
	}
	if s.err != nil {
		return step.AssistantEvent{Type: step.EventError, Err: s.err.Error()}, s.err
	}

	for {
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return step.AssistantEvent{Type: step.EventError, Err: s.err.Error()}, s.err
		default:
		}

		if !s.stream.Next() {
			if err := s.stream.Err(); err != nil {
				s.err = err
				return step.AssistantEvent{Type: step.EventError, Err: err.Error()}, err
			}
			// Stream ended
			return s.finalize()
		}

		chunk := s.stream.Current()
		s.processChunk(chunk)
		if len(s.pendingEvents) > 0 {
			return s.dequeuePendingEvent()
		}
	}
}

func (s *Stream) enqueue(ev step.AssistantEvent) {
	s.pendingEvents = append(s.pendingEvents, ev)
}

func (s *Stream) dequeuePendingEvent() (step.AssistantEvent, error) {
	ev := s.pendingEvents[0]
	s.pendingEvents = s.pendingEvents[1:]
	if s.debug != nil {
		rec := base.NewDebugRecord("event", ev)
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	if ev.Type == step.EventDone {
		return ev, io.EOF
	}
	return ev, nil
}

func (s *Stream) processChunk(chunk openai.ChatCompletionChunk) {
	if s.debug != nil {
		rec := base.NewDebugRecord("chunk", chunk)
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	// Extract usage if present
	if chunk.Usage.TotalTokens > 0 {
		s.usage = &step.Usage{
			InputTokens:  int(chunk.Usage.PromptTokens),
			OutputTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:  int(chunk.Usage.TotalTokens),
		}
		// Handle cached tokens if available
		if chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
			s.usage.CachedReadTokens = int(chunk.Usage.PromptTokensDetails.CachedTokens)
		}
	}

	if len(chunk.Choices) == 0 {
		return
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Check finish reason
	if choice.FinishReason != "" {
		s.stopReason = mapFinishReason(string(choice.FinishReason))
	}

	// Try to extract reasoning using handler
	if s.reasoningHandler != nil {
		deltaMap := deltaToMap(delta)
		if text, isThinking := s.reasoningHandler.ExtractThinking(deltaMap); isThinking {
			if s.stage != stageThinking {
				s.stage = stageThinking
				s.enqueue(step.AssistantEvent{Type: step.EventThinkingStart})
			}
			s.enqueue(step.AssistantEvent{
				Type:  step.EventThinkingDelta,
				Delta: text,
			})
			return
		}
	}

	// Process text content
	if delta.Content != "" {
		if s.stage != stageText {
			s.emitStageEnd()
			s.stage = stageText
			s.enqueue(step.AssistantEvent{Type: step.EventTextStart})
		}
		s.textContent = append(s.textContent, delta.Content)
		s.enqueue(step.AssistantEvent{
			Type:  step.EventTextDelta,
			Delta: delta.Content,
		})
		return
	}

	// Process tool calls
	for _, tc := range delta.ToolCalls {
		idx := int(tc.Index)

		// If switching to a different tool call, emit end for the previous one
		if s.hasCurrentTool && s.currentToolIdx != idx {
			s.emitToolCallEnd(s.currentToolIdx)
		}

		// Initialize accumulator if needed
		if _, exists := s.toolCalls[idx]; !exists {
			s.toolCalls[idx] = &toolCallAccumulator{}
		}
		acc := s.toolCalls[idx]

		// Update accumulator
		if tc.ID != "" {
			acc.id = tc.ID
		}
		if tc.Function.Name != "" {
			acc.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			acc.argsStr += tc.Function.Arguments
		}

		// Emit tool call start if we have id and name
		if !s.emittedToolStart[idx] && acc.id != "" && acc.name != "" {
			s.emittedToolStart[idx] = true
			s.currentToolIdx = idx
			s.hasCurrentTool = true

			// Only emit stage end when transitioning from non-tool stage
			if s.stage != stageTool {
				s.emitStageEnd()
				s.stage = stageTool
			}

			s.enqueue(step.AssistantEvent{
				Type: step.EventToolCallStart,
				ToolCall: &step.ToolCallPart{
					CallID: acc.id,
					Name:   acc.name,
				},
			})
		}

		// Emit tool call delta for arguments
		if tc.Function.Arguments != "" && s.emittedToolStart[idx] {
			s.enqueue(step.AssistantEvent{
				Type:  step.EventToolCallDelta,
				Delta: tc.Function.Arguments,
			})
		}
	}
}

func (s *Stream) emitStageEnd() {
	switch s.stage {
	case stageThinking:
		if thinkingParts := s.reasoningHandler.FlushThinking(); len(thinkingParts) > 0 {
			for _, part := range thinkingParts {
				s.parts = append(s.parts, part)
			}
		}
		s.enqueue(step.AssistantEvent{Type: step.EventThinkingEnd})
	case stageText:
		s.flushText()
		s.enqueue(step.AssistantEvent{Type: step.EventTextEnd})
	case stageTool:
		s.emitToolCallEnds()
	}
}

func (s *Stream) emitToolCallEnd(idx int) {
	acc, exists := s.toolCalls[idx]
	if !exists || !s.emittedToolStart[idx] {
		return
	}
	delete(s.emittedToolStart, idx)
	s.enqueue(step.AssistantEvent{
		Type: step.EventToolCallEnd,
		ToolCall: &step.ToolCallPart{
			CallID:   acc.id,
			Name:     acc.name,
			ArgsJSON: json.RawMessage(acc.argsStr),
		},
	})
}

func (s *Stream) emitToolCallEnds() {
	if s.hasCurrentTool {
		s.emitToolCallEnd(s.currentToolIdx)
		s.hasCurrentTool = false
	}
}

func (s *Stream) finalize() (step.AssistantEvent, error) {
	s.done = true

	// Emit End event for current stage
	s.emitStageEnd()

	// Flush tool calls to parts
	for _, acc := range s.toolCalls {
		if acc.id != "" && acc.name != "" {
			s.parts = append(s.parts, step.ToolCallPart{
				CallID:   acc.id,
				Name:     acc.name,
				ArgsJSON: json.RawMessage(acc.argsStr),
			})
		}
	}

	if s.stopReason == "" {
		s.stopReason = step.StopStop
	}

	// Enqueue Done event
	s.enqueue(step.AssistantEvent{
		Type:   step.EventDone,
		Reason: s.stopReason,
	})

	// Return first pending event
	return s.dequeuePendingEvent()
}

func (s *Stream) flushText() {
	if len(s.textContent) > 0 {
		text := ""
		for _, t := range s.textContent {
			text += t
		}
		s.parts = append(s.parts, step.TextPart{Text: text})
		s.textContent = nil
	}
}

// Result returns the final generation result.
func (s *Stream) Result() (*step.GenerateResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.done {
		return nil, errors.New("stream not finished")
	}
	if s.err != nil {
		return nil, s.err
	}

	msg := step.AssistantMessage{Parts: s.parts}
	return &step.GenerateResult{
		Message:    msg,
		Usage:      s.usage,
		StopReason: s.stopReason,
	}, nil
}

// Close closes the stream.
func (s *Stream) Close() error {
	if s.debug != nil {
		_ = s.debug.Close()
	}
	return s.stream.Close()
}

func mapFinishReason(reason string) step.StopReason {
	switch reason {
	case "stop":
		return step.StopStop
	case "length":
		return step.StopLength
	case "tool_calls":
		return step.StopToolUse
	default:
		return step.StopStop
	}
}

func deltaToMap(delta openai.ChatCompletionChunkChoiceDelta) map[string]any {
	data, _ := json.Marshal(delta)
	var m map[string]any
	_ = json.Unmarshal(data, &m)
	return m
}
