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

package master

import (
	"errors"
	"time"

	"github.com/gorse-io/gorse/dataset"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func (s *MasterTestSuite) TestRunTaskRecoversPanic() {
	panicsBefore := metricValue(TaskPanicsTotalVec.WithLabelValues("boom"))
	failuresBefore := metricValue(TaskFailuresTotalVec.WithLabelValues("boom"))

	err := s.runTask("boom", func() error {
		var shape []int
		_ = shape[2] // tensor shape mismatch style runtime panic
		return nil
	})
	s.Error(err)
	s.Contains(err.Error(), "task boom panicked")
	s.Equal(panicsBefore+1, metricValue(TaskPanicsTotalVec.WithLabelValues("boom")))
	s.Equal(failuresBefore+1, metricValue(TaskFailuresTotalVec.WithLabelValues("boom")))

	err = s.runTask("boom", func() error { return errors.New("failed") })
	s.Error(err)
	s.Equal(failuresBefore+2, metricValue(TaskFailuresTotalVec.WithLabelValues("boom")))

	s.NoError(s.runTask("boom", func() error { return nil }))
	s.Greater(metricValue(TaskLastSuccessTimestampVec.WithLabelValues("boom")), float64(0))
	s.GreaterOrEqual(metricValue(TaskDurationSecondsVec.WithLabelValues("boom")), float64(0))
}

func (s *MasterTestSuite) TestCollectVectorStoreStats() {
	ctx := s.T().Context()
	collection := vectors.ItemToItemCollection("stats")
	s.NoError(s.VectorClient.AddCollection(ctx, collection, 4, vectors.Euclidean, vectors.VectorConfig{}))
	s.NoError(s.VectorClient.AddVectors(ctx, collection, []vectors.Vector{
		{Id: "a", Values: []float32{1, 0, 0, 0}},
		{Id: "b", Values: []float32{0, 1, 0, 0}},
	}))
	s.NoError(s.collectVectorStoreStats(ctx))
	s.Equal(float64(2), metricValue(VectorStoreVectorsTotalVec.WithLabelValues(collection)))
	s.Equal(float64(2*4*4), metricValue(VectorStoreEstimatedBytesVec.WithLabelValues(collection)))
}

func (s *MasterTestSuite) TestCollectGarbageCountsDocuments() {
	ctx := s.T().Context()
	s.NoError(s.CacheClient.AddScores(ctx, cache.Recommend, "u1", []cache.Score{{Id: "a", Score: 1}, {Id: "b", Score: 2}}))
	s.NoError(s.CacheClient.AddScores(ctx, cache.Recommend, "u2", []cache.Score{{Id: "a", Score: 1}}))
	s.NoError(s.CacheClient.AddScores(ctx, cache.NonPersonalized, "popular", []cache.Score{{Id: "a", Score: 1}}))
	s.Config.Recommend.NonPersonalized = nil
	s.NoError(s.collectGarbage(ctx, dataset.NewDataset(time.Now(), 0, 0)))
	s.Equal(float64(3), metricValue(CacheDocumentsTotalVec.WithLabelValues(cache.Recommend)))
	s.Equal(float64(1), metricValue(CacheDocumentsTotalVec.WithLabelValues(cache.NonPersonalized+"/popular")))
}

// metricValue reads the current value of a counter or gauge without pulling
// prometheus/testutil (and its extra module requirements) into go.mod.
func metricValue(m prometheus.Metric) float64 {
	var out dto.Metric
	if err := m.Write(&out); err != nil {
		panic(err)
	}
	if out.Counter != nil {
		return out.Counter.GetValue()
	}
	return out.Gauge.GetValue()
}
