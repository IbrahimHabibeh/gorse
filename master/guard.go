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

// VideoHub fork: task guards and observability for the master.
//
// Every step of the periodic task cycle runs through runTask so that a panic
// in one model (for example a tensor shape mismatch during training) fails
// that task cleanly, is counted and leaves the task loop alive. The same hook
// records durations, last-success timestamps and failures per task.

package master

import (
	"context"
	"time"

	"github.com/gorse-io/gorse/common/log"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/pkg/errors"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

var (
	TaskDurationSecondsVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "task_duration_seconds",
		Help:      "Duration of the last run of each master task.",
	}, []string{"task"})
	TaskLastSuccessTimestampVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "task_last_success_timestamp_seconds",
		Help:      "Unix time of the last successful run of each master task.",
	}, []string{"task"})
	TaskFailuresTotalVec = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "task_failures_total",
		Help:      "Failed runs of each master task, including recovered panics.",
	}, []string{"task"})
	TaskPanicsTotalVec = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "task_panics_total",
		Help:      "Panics recovered inside master tasks.",
	}, []string{"task"})
	ModelIdVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "model_id",
		Help:      "Generation id (unix milliseconds) of the current model.",
	}, []string{"model"})
	ItemToItemVectorsTotalVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "item_to_item_vectors_total",
		Help:      "Vectors stored in each item-to-item collection after the last job.",
	}, []string{"recommender"})
	ItemToItemItemsWithoutVectorVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "item_to_item_items_without_vector",
		Help:      "Items skipped by the last item-to-item job because the source column was empty.",
	}, []string{"recommender"})
	ItemToItemInvalidVectorsVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "item_to_item_invalid_vectors",
		Help:      "Items dropped by the last item-to-item job because their vector dimension did not match the collection.",
	}, []string{"recommender"})
	ItemToItemLastUpdateTimestampVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "item_to_item_last_update_timestamp_seconds",
		Help:      "Unix time of the last completed item-to-item job per recommender.",
	}, []string{"recommender"})
	CTRMalformedEmbeddings = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "ctr_dataset_malformed_embeddings",
		Help:      "Item embeddings dropped from the last click-through rate dataset because their dimension was not the majority dimension.",
	})
	CacheDocumentsTotalVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "cache_documents_total",
		Help:      "Cached score documents per collection observed by the last garbage collection scan.",
	}, []string{"collection"})
	CacheMemoryBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "cache_memory_bytes",
		Help:      "Memory used by the cache store as reported by the backend (Redis used_memory).",
	})
	VectorStoreVectorsTotalVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "vector_store_vectors_total",
		Help:      "Vectors stored per collection in the vector store.",
	}, []string{"collection"})
	VectorStoreEstimatedBytesVec = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "gorse",
		Subsystem: "master",
		Name:      "vector_store_estimated_bytes",
		Help:      "Lower-bound memory estimate per dense collection (vectors * dimension * 4 bytes); excludes index structures.",
	}, []string{"collection"})
)

// runTask runs one step of the task cycle with a panic boundary and records
// its duration, last success time and failures.
func (m *Master) runTask(name string, fn func() error) (err error) {
	start := time.Now()
	defer func() {
		if r := recover(); r != nil {
			TaskPanicsTotalVec.WithLabelValues(name).Inc()
			err = errors.Errorf("task %s panicked: %v", name, r)
			log.Logger().Error("task panicked",
				zap.String("task", name),
				zap.Any("panic", r),
				zap.Stack("stack"))
		}
		TaskDurationSecondsVec.WithLabelValues(name).Set(time.Since(start).Seconds())
		if err != nil {
			TaskFailuresTotalVec.WithLabelValues(name).Inc()
		} else {
			TaskLastSuccessTimestampVec.WithLabelValues(name).Set(float64(time.Now().Unix()))
		}
	}()
	return fn()
}

// collectVectorStoreStats publishes vector counts and a memory lower bound for
// every collection so unbounded growth is visible before it exhausts memory.
func (m *Master) collectVectorStoreStats(ctx context.Context) error {
	collections, err := m.VectorClient.ListCollections(ctx)
	if err != nil {
		return errors.WithStack(err)
	}
	VectorStoreVectorsTotalVec.Reset()
	VectorStoreEstimatedBytesVec.Reset()
	for _, collection := range collections {
		count, err := m.VectorClient.CountVectors(ctx, collection)
		if err != nil {
			log.Logger().Warn("failed to count vectors", zap.String("collection", collection), zap.Error(err))
			continue
		}
		VectorStoreVectorsTotalVec.WithLabelValues(collection).Set(float64(count))
		info, err := m.VectorClient.DescribeCollection(ctx, collection)
		if err != nil || info == nil || info.Dimension <= 0 {
			continue
		}
		VectorStoreEstimatedBytesVec.WithLabelValues(collection).Set(float64(count) * float64(info.Dimension) * 4)
	}
	return nil
}

// collectCacheStats publishes the cache backend's memory usage when the
// backend can report it (Redis).
func (m *Master) collectCacheStats(ctx context.Context) error {
	reporter, ok := m.CacheClient.(cache.MemoryReporter)
	if !ok {
		return nil
	}
	used, err := reporter.MemoryUsage(ctx)
	if err != nil {
		return errors.WithStack(err)
	}
	CacheMemoryBytes.Set(float64(used))
	return nil
}

// cacheDocumentLabel groups scanned cache documents into the collection label
// used by CacheDocumentsTotalVec: non-personalized recommenders are reported
// per recommender, everything else per collection.
func cacheDocumentLabel(collection, subset string) string {
	if collection == cache.NonPersonalized && subset != "" {
		return collection + "/" + subset
	}
	return collection
}

// publishItemToItemStats records vector counts and skipped items after an
// item-to-item job. Recommenders expose their writer statistics through the
// optional VectorWriterStats method.
func (m *Master) publishItemToItemStats(ctx context.Context, name string, recommender any) {
	if stats, ok := recommender.(interface {
		VectorWriterStats() (added, skipped, invalid int)
	}); ok {
		_, skipped, invalid := stats.VectorWriterStats()
		ItemToItemItemsWithoutVectorVec.WithLabelValues(name).Set(float64(skipped))
		ItemToItemInvalidVectorsVec.WithLabelValues(name).Set(float64(invalid))
	}
	if count, err := m.VectorClient.CountVectors(ctx, vectors.ItemToItemCollection(name)); err == nil {
		ItemToItemVectorsTotalVec.WithLabelValues(name).Set(float64(count))
	}
	ItemToItemLastUpdateTimestampVec.WithLabelValues(name).Set(float64(time.Now().Unix()))
}
