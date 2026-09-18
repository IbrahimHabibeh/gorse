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

package worker

import (
	"time"

	"github.com/gorse-io/gorse/common/expression"
	"github.com/gorse-io/gorse/model/ctr"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/samber/lo"
)

// panickingFactorizationMachine simulates a tensor shape mismatch inside the
// click-through rate model, which upstream surfaces as a runtime panic.
type panickingFactorizationMachine struct {
	mockFactorizationMachine
}

func (m panickingFactorizationMachine) BatchPredict(_ []lo.Tuple4[string, string, []ctr.Label, []ctr.Label], _ [][]ctr.Embedding, _ int) []float32 {
	var indices []float32
	_ = indices[3] // index out of range, like a feature vector longer than the trained dimension
	return nil
}

func (m panickingFactorizationMachine) BatchInternalPredict(_ []lo.Tuple2[[]int32, []float32], _ [][][]uint16, _ int) []float32 {
	panic("unexpected call")
}

func (suite *WorkerTestSuite) TestRecommendSurvivesRankingPanic() {
	ctx := suite.T().Context()
	suite.Config.Recommend.Collaborative.Type = "none"
	suite.Config.Recommend.Ranker.Type = "fm"
	suite.Config.Recommend.Ranker.Recommenders = []string{"latest"}
	suite.Config.Recommend.DataSource.PositiveFeedbackTypes = []expression.FeedbackTypeExpression{expression.MustParseFeedbackTypeExpression("a")}
	suite.Config.Recommend.CacheSize = 3
	suite.ClickThroughRateModel = new(panickingFactorizationMachine)

	suite.NoError(suite.DataClient.BatchInsertItems(ctx, []data.Item{
		{ItemId: "0", Timestamp: time.Unix(0, 0)},
		{ItemId: "1", Timestamp: time.Unix(1, 0)},
		{ItemId: "2", Timestamp: time.Unix(2, 0)},
		{ItemId: "3", Timestamp: time.Unix(3, 0)},
	}))
	suite.NoError(suite.DataClient.BatchInsertFeedback(ctx, []data.Feedback{
		{FeedbackKey: data.FeedbackKey{FeedbackType: "a", UserId: "0", ItemId: "0"}},
	}, true, true, true))

	panicsBefore := metricValue(RecommendPanicsTotal)
	failuresBefore := metricValue(RankingFailuresTotal.WithLabelValues("panic"))

	// A panicking ranker must neither crash the worker nor leave the user without recommendations.
	suite.Recommend(ctx, []data.User{{UserId: "0"}}, nil)

	suite.Equal(panicsBefore, metricValue(RecommendPanicsTotal), "the panic must be contained by the ranking boundary")
	suite.Equal(failuresBefore+1, metricValue(RankingFailuresTotal.WithLabelValues("panic")))
	recommends, err := suite.CacheClient.SearchScores(ctx, cache.Recommend, "0", nil, 0, 5)
	suite.NoError(err)
	suite.Equal([]string{"3", "2", "1"}, lo.Map(recommends, func(score cache.Score, _ int) string { return score.Id }),
		"unranked candidates are served when ranking fails")
}

func (suite *WorkerTestSuite) TestRecommendSurvivesJobPanic() {
	ctx := suite.T().Context()
	suite.Config.Recommend.Collaborative.Type = "none"
	suite.Config.Recommend.Ranker.Recommenders = []string{"latest"}
	suite.Config.Recommend.DataSource.PositiveFeedbackTypes = []expression.FeedbackTypeExpression{expression.MustParseFeedbackTypeExpression("a")}
	suite.Config.Recommend.CacheSize = 3
	// Provoke a panic outside the ranking boundary: the pipeline inspects the
	// model before ranking, so a corrupt model object hits the per-user job
	// boundary instead.
	suite.Config.Recommend.Ranker.Type = "fm"
	suite.ClickThroughRateModel = new(brokenFactorizationMachine)

	suite.NoError(suite.DataClient.BatchInsertItems(ctx, []data.Item{
		{ItemId: "0", Timestamp: time.Unix(0, 0)},
		{ItemId: "1", Timestamp: time.Unix(1, 0)},
	}))
	suite.NoError(suite.DataClient.BatchInsertFeedback(ctx, []data.Feedback{
		{FeedbackKey: data.FeedbackKey{FeedbackType: "a", UserId: "0", ItemId: "0"}},
		{FeedbackKey: data.FeedbackKey{FeedbackType: "a", UserId: "1", ItemId: "0"}},
	}, true, true, true))

	panicsBefore := metricValue(RecommendPanicsTotal)
	suite.Recommend(ctx, []data.User{{UserId: "0"}, {UserId: "1"}}, nil)
	suite.Equal(panicsBefore+2, metricValue(RecommendPanicsTotal), "one recovered panic per user job")
}

// brokenFactorizationMachine panics as soon as the pipeline inspects it, i.e.
// outside the ranking boundary, exercising the per-user job boundary.
type brokenFactorizationMachine struct {
	mockFactorizationMachine
}

func (m brokenFactorizationMachine) Invalid() bool {
	panic("model state is corrupt")
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
