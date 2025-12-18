package base

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

// DebugLogger writes JSON objects as JSONL.
// It is safe for concurrent use.
type DebugLogger struct {
	mu  sync.Mutex
	f   *os.File
	enc *json.Encoder
}

// NewDebugLogger creates a new debug logger that writes to the specified path.
// If path is empty, returns nil (debug logging disabled).
func NewDebugLogger(path string) (*DebugLogger, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &DebugLogger{f: f, enc: json.NewEncoder(f)}, nil
}

func (l *DebugLogger) Close() error {
	if l == nil || l.f == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.f.Close()
}

// Log writes a JSON line.
func (l *DebugLogger) Log(v any) error {
	if l == nil || l.enc == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.enc.Encode(v)
}

// DebugRecord is a normalized JSONL entry.
type DebugRecord struct {
	Time     string `json:"time"`
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
	Type     string `json:"type"`
	Data     any    `json:"data,omitempty"`
}

func NewDebugRecord(recordType string, data any) DebugRecord {
	return DebugRecord{
		Time: time.Now().UTC().Format(time.RFC3339Nano),
		Type: recordType,
		Data: data,
	}
}
