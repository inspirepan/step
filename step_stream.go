package step

import (
	"context"
	"errors"
	"io"
	"sync"
)

// StepStream exposes streaming access to a single step.
type StepStream interface {
	Next(ctx context.Context) (StepEvent, error)
	Result() (*StepResult, error)
	Cancel()
	Close() error
}

type stepStream struct {
	ctx    context.Context
	cancel context.CancelFunc

	events chan StepEvent

	result    StepResult
	resultErr error
	done      chan struct{}

	mu sync.Mutex
}

// StepStreamed runs a step and returns a stream of events.
func StepStreamed(parent context.Context, req StepRequest) (StepStream, error) {
	if req.Provider == nil {
		return nil, ErrNoProvider
	}

	ctx, cancel := context.WithCancel(parent)
	s := &stepStream{
		ctx:    ctx,
		cancel: cancel,
		events: make(chan StepEvent, 16),
		done:   make(chan struct{}),
	}

	go s.run(req)
	return s, nil
}

func (s *stepStream) Next(ctx context.Context) (StepEvent, error) {
	select {
	case <-ctx.Done():
		return StepEvent{}, ctx.Err()
	case ev, ok := <-s.events:
		if !ok {
			return StepEvent{}, io.EOF
		}
		return ev, nil
	}
}

func (s *stepStream) Result() (*StepResult, error) {
	<-s.done
	return &s.result, s.resultErr
}

func (s *stepStream) Cancel() {
	s.cancel()
}

func (s *stepStream) Close() error {
	s.cancel()
	<-s.done
	return nil
}

func (s *stepStream) run(req StepRequest) {
	defer close(s.events)
	defer close(s.done)
	defer s.cancel()

	s.emit(StepEvent{Type: StepEventStart})

	genReq := GenerateRequest{
		SystemPrompt: req.SystemPrompt,
		History:      req.History,
		Tools:        collectToolSpecs(req.Tools),
	}

	stream, err := req.Provider.GenerateStream(s.ctx, genReq)
	if err != nil {
		s.finish(nil, err)
		return
	}
	defer stream.Close()

	for {
		ev, err := stream.Next(s.ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			s.finish(nil, err)
			return
		}
		s.emit(StepEvent{Type: StepEventAssistant, Assistant: &ev})
	}

	genRes, err := stream.Result()
	if err != nil {
		s.finish(nil, err)
		return
	}

	result := StepResult{Assistant: *genRes}
	assistantMsg := genRes.Message.(AssistantMessage)
	toolCalls := extractToolCalls(assistantMsg)
	result.ToolCalls = toolCalls

	toolResults, cancelled := s.executeTools(toolCalls, req.Tools)
	result.ToolResults = toolResults
	if cancelled || s.ctx.Err() != nil {
		result.Cancelled = true
	}

	result.NewMessages = append(result.NewMessages, genRes.Message)
	result.NewMessages = append(result.NewMessages, toolResultsToMessages(toolResults)...)

	s.finish(&result, nil)
}

func (s *stepStream) finish(res *StepResult, err error) {
	if res != nil {
		s.result = *res
	}
	s.addError(err)
	if res != nil {
		s.emit(StepEvent{Type: StepEventEnd, Final: res})
	}
}

func (s *stepStream) emit(ev StepEvent) {
	select {
	case <-s.ctx.Done():
		return
	case s.events <- ev:
	}
}

func (s *stepStream) addError(err error) {
	if err == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.resultErr == nil {
		s.resultErr = err
		return
	}
	s.resultErr = errors.Join(s.resultErr, err)
}
