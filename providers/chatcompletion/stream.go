package chatcompletion

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/inspirepan/step"
	"github.com/inspirepan/step/providers/base"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

type streamStage int

const (
	stageWaiting streamStage = iota
	stageThinking
	stageText
	stageTool
)

// Stream implements step.ProviderStream for OpenAI Chat Completions API.
type Stream struct {
	providerName string
	modelName    string
	stream       *ssestream.Stream[openai.ChatCompletionChunk]
	debug        *base.DebugLogger

	reasoningHandler ReasoningHandler

	mu sync.Mutex

	stage streamStage
	done  bool
	err   error

	pending []step.ProviderUpdate

	// Accumulators
	textContent []string
	toolCalls   map[int]*toolCallAccumulator

	stopReason step.StopReason
	usage      *step.Usage
	parts      []step.Part
}

type toolCallAccumulator struct {
	id      string
	name    string
	argsStr string
}

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
		debug:            debug,
		reasoningHandler: handler,
		stage:            stageWaiting,
		toolCalls:        make(map[int]*toolCallAccumulator),
	}
}

func (s *Stream) Next(ctx context.Context) (step.ProviderUpdate, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pending) > 0 {
		return s.dequeue()
	}
	if s.done {
		return nil, io.EOF
	}
	if s.err != nil {
		return nil, s.err
	}

	for {
		select {
		case <-ctx.Done():
			s.err = ctx.Err()
			return nil, s.err
		default:
		}

		if !s.stream.Next() {
			if err := s.stream.Err(); err != nil {
				s.err = err
				return nil, s.err
			}
			s.finalize()
			if len(s.pending) > 0 {
				return s.dequeue()
			}
			return nil, io.EOF
		}

		chunk := s.stream.Current()
		s.processChunk(chunk)
		if len(s.pending) > 0 {
			return s.dequeue()
		}
	}
}

func (s *Stream) Close() error {
	if s.debug != nil {
		_ = s.debug.Close()
	}
	return s.stream.Close()
}

func (s *Stream) enqueue(up step.ProviderUpdate) {
	s.pending = append(s.pending, up)
}

func (s *Stream) dequeue() (step.ProviderUpdate, error) {
	up := s.pending[0]
	s.pending = s.pending[1:]

	if s.debug != nil {
		rec := base.NewDebugRecord("update", up)
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	return up, nil
}

func (s *Stream) processChunk(chunk openai.ChatCompletionChunk) {
	if s.debug != nil {
		rec := base.NewDebugRecord("chunk", chunk)
		rec.Provider = s.providerName
		rec.Model = s.modelName
		_ = s.debug.Log(rec)
	}

	// Usage
	if chunk.Usage.TotalTokens > 0 {
		s.usage = &step.Usage{
			InputTokens:  int(chunk.Usage.PromptTokens),
			OutputTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:  int(chunk.Usage.TotalTokens),
		}
		if chunk.Usage.PromptTokensDetails.CachedTokens > 0 {
			s.usage.CachedReadTokens = int(chunk.Usage.PromptTokensDetails.CachedTokens)
		}
	}

	if len(chunk.Choices) == 0 {
		return
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	if choice.FinishReason != "" {
		s.stopReason = mapFinishReason(string(choice.FinishReason))
	}

	// Thinking
	if s.reasoningHandler != nil {
		deltaMap := deltaToMap(delta)
		if text, isThinking := s.reasoningHandler.ExtractThinking(deltaMap); isThinking {
			if s.stage != stageThinking {
				s.emitStageEnd()
				s.stage = stageThinking
			}
			s.enqueue(step.ProviderDeltaUpdate{Delta: step.ThinkingDelta{Delta: text}})
			return
		}
	}

	// Text
	if delta.Content != "" {
		if s.stage != stageText {
			s.emitStageEnd()
			s.stage = stageText
		}
		s.textContent = append(s.textContent, delta.Content)
		s.enqueue(step.ProviderDeltaUpdate{Delta: step.TextDelta{Delta: delta.Content}})
		return
	}

	// Tool calls
	for _, tc := range delta.ToolCalls {
		idx := int(tc.Index)
		if s.stage != stageTool {
			s.emitStageEnd()
			s.stage = stageTool
		}
		if _, exists := s.toolCalls[idx]; !exists {
			s.toolCalls[idx] = &toolCallAccumulator{}
		}
		acc := s.toolCalls[idx]
		if tc.ID != "" {
			acc.id = tc.ID
		}
		if tc.Function.Name != "" {
			acc.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			acc.argsStr += tc.Function.Arguments
			s.enqueue(step.ProviderDeltaUpdate{Delta: step.ToolCallDelta{CallID: acc.id, Name: acc.name, ArgsDelta: tc.Function.Arguments}})
		}
	}
}

func (s *Stream) finalize() {
	s.done = true

	if s.stopReason == "" {
		s.stopReason = step.StopStop
	}

	s.emitStageEnd()

	for _, acc := range s.toolCalls {
		if acc.id == "" || acc.name == "" {
			continue
		}
		s.parts = append(s.parts, step.ToolCallPart{
			CallID:   acc.id,
			Name:     acc.name,
			ArgsJSON: json.RawMessage(acc.argsStr),
		})
	}

	msg := step.AssistantMessage{
		Parts:      s.parts,
		Timestamp:  time.Now().UnixMilli(),
		Usage:      s.usage,
		StopReason: s.stopReason,
	}
	s.enqueue(step.ProviderMessageUpdate{Message: msg})
}

func (s *Stream) emitStageEnd() {
	switch s.stage {
	case stageThinking:
		if thinkingParts := s.reasoningHandler.FlushThinking(); len(thinkingParts) > 0 {
			for _, part := range thinkingParts {
				s.parts = append(s.parts, part)
			}
		}
	case stageText:
		s.flushText()
	}
}

func (s *Stream) flushText() {
	if len(s.textContent) == 0 {
		return
	}
	text := ""
	for _, t := range s.textContent {
		text += t
	}
	s.parts = append(s.parts, step.TextPart{Text: text})
	s.textContent = nil
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

var _ step.ProviderStream = (*Stream)(nil)
