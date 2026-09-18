// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VideoHub fork: panic boundaries and additional metrics for the worker.
//
// Model inference runs on user-supplied label shapes. A malformed sample must
// degrade one user's recommendation, never terminate the process, so every
// per-user job and every ranking call recovers from panics, reports them and
// continues.

package worker

import (
	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/pkg/errors"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/samber/lo"
	"go.uber.org/zap"
)

var (
	RecommendPanicsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "worker",
		Name:      "recommend_panics_total",
		Help:      "Panics recovered while generating recommendations for a user.",
	})
	RankingFailuresTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "worker",
		Name:      "ranking_failures_total",
		Help:      "Ranking calls that failed and fell back to unranked candidates.",
	}, []string{"reason"})
	RecommendOutcomesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "worker",
		Name:      "recommend_outcomes_total",
		Help:      "Per-user outcomes of the offline recommendation pipeline.",
	}, []string{"outcome"})
	ModelIdVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "worker",
		Name:      "model_id",
		Help:      "Generation id (unix milliseconds) of the model currently loaded by this worker.",
	}, []string{"model"})
)

// safeBatchPredict runs BatchPredict inside a panic boundary. Tensor shape
// mismatches surface as panics deep in the model; they are converted into an
// error carrying the user id so the caller can degrade gracefully.
func safeBatchPredict(
	predictor ctr.BatchInference,
	userId string,
	inputs []lo.Tuple4[string, string, []ctr.Label, []ctr.Label],
	embeddings [][]ctr.Embedding,
	jobs int,
) (output []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			RankingFailuresTotal.WithLabelValues("panic").Inc()
			err = errors.Errorf("ranking model panicked for user %s: %v", userId, r)
			log.Logger().Error("ranking model panicked",
				zap.String("user_id", userId),
				zap.Int("n_candidates", len(inputs)),
				zap.Any("panic", r),
				zap.Stack("stack"))
		}
	}()
	output = predictor.BatchPredict(inputs, embeddings, jobs)
	if len(output) != len(inputs) {
		RankingFailuresTotal.WithLabelValues("output_size").Inc()
		return nil, errors.Errorf("ranking model returned %d scores for %d candidates", len(output), len(inputs))
	}
	return output, nil
}

// recoverUserJob converts a panic inside a per-user recommendation job into a
// logged, counted failure so the remaining users are still processed.
func recoverUserJob(userId string) {
	if r := recover(); r != nil {
		RecommendPanicsTotal.Inc()
		RecommendOutcomesTotal.WithLabelValues("panic").Inc()
		log.Logger().Error("panic recovered while recommending for user",
			zap.String("user_id", userId),
			zap.Any("panic", r),
			zap.Stack("stack"))
	}
}
