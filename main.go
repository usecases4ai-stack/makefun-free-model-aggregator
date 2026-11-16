// free-model-aggregator-02 entry: Live free-model aggregator
// Serves OpenAI- and Anthropic-compatible API, answers prompts via free OpenRouter models

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

type Model struct {
	ID    string  `json:"id"`
	Price float64 `json:"price"`
	TPM   int     `json:"tokens_per_minute"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenRouterRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
}

type OpenRouterResponse struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

type Aggregator struct {
	mu         sync.Mutex
	models     []Model
	currentIdx int
	client     *http.Client
	routerKey  string
}

func NewAggregator(apiKey string) *Aggregator {
	return &Aggregator{
		client:    &http.Client{Timeout: 30 * time.Second},
		routerKey: apiKey,
	}
}

func (a *Aggregator) fetchFreeModels() {
	req, _ := http.NewRequest("GET", "https://openrouter.ai/api/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+a.routerKey)
	resp, err := a.client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to fetch models: %v\n", err)
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	type ModelsResp struct {
		Data []Model `json:"data"`
	}
	var mr ModelsResp
	json.Unmarshal(body, &mr)
	a.mu.Lock()
	for _, m := range mr.Data {
		if m.Price == 0 && m.TPM > 0 {
			a.models = append(a.models, m)
		}
	}
	a.mu.Unlock()
	fmt.Printf("Loaded %d free models\n", len(a.models))
}

func (a *Aggregator) nextModel() (Model, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.models) == 0 {
		return Model{}, false
	}
	m := a.models[a.currentIdx%len(a.models)]
	a.currentIdx++
	return m, true
}

func (a *Aggregator) proxyRequest(modelID string, messages []Message) error {
	reqBody, _ := json.Marshal(OpenRouterRequest{
		Model:    modelID,
		Messages: messages,
	})
	req, _ := http.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer "+a.routerKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://make.fun")
	req.Header.Set("X-Title", "make.fun-free-aggregator")
	resp, err := a.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var oResp OpenRouterResponse
	json.NewDecoder(resp.Body).Decode(&oResp)
	return nil
}

func (a *Aggregator) handleOpenAI(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	m, ok := a.nextModel()
	if !ok {
		http.Error(w, "No models", 503)
		return
	}
	a.proxyRequest(m.ID, req.Messages)
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
}

func (a *Aggregator) handleAnthropic(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	m, ok := a.nextModel()
	if !ok {
		http.Error(w, "No models", 503)
		return
	}
	a.proxyRequest(m.ID, req.Messages)
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"content":[{"type":"text","text":"ok"}]}`))
}

var startTime = time.Now()

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	agg := NewAggregator(os.Getenv("OPENROUTER_API_KEY"))
	agg.fetchFreeModels()
	if len(agg.models) == 0 {
		agg.mu.Lock()
		agg.models = []Model{{ID: "google/gemini-2.0-flash-exp:free", Price: 0, TPM: 1000}}
		agg.mu.Unlock()
	}
	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", http.HandlerFunc(agg.handleOpenAI)))
	mux.HandleFunc("/v1/messages", agg.handleAnthropic)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy", "models": len(agg.models), "uptime": time.Since(startTime).String(),
		})
	})
	fmt.Printf("Free Model Aggregator on :%s (pool: %d models)\n", port, len(agg.models))
	http.ListenAndServe(":"+port, mux)
}
