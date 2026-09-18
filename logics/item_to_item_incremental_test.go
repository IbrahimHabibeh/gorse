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

package logics

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/vectors"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newIncrementalFixture(t *testing.T) (vectors.Database, data.Database) {
	vectorClient, err := vectors.Open(fmt.Sprintf("xvec://%s/vectors", t.TempDir()), "")
	require.NoError(t, err)
	require.NoError(t, vectorClient.Init())
	dataClient, err := data.Open(fmt.Sprintf("sqlite://%s/data.db", t.TempDir()), "")
	require.NoError(t, err)
	require.NoError(t, dataClient.Init())
	t.Cleanup(func() {
		assert.NoError(t, vectorClient.Close())
		assert.NoError(t, dataClient.Close())
	})
	return vectorClient, dataClient
}

func TestItemEmbedding(t *testing.T) {
	cfg := config.ItemToItemConfig{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"}

	t.Run("json numbers from the REST layer", func(t *testing.T) {
		item := &data.Item{ItemId: "a", Labels: map[string]any{"embedding": []any{json.Number("0.5"), json.Number("1")}}}
		embedding, present, err := ItemEmbedding(cfg, item)
		require.NoError(t, err)
		assert.True(t, present)
		assert.Equal(t, []float32{0.5, 1}, embedding)
	})
	t.Run("float64 from the data store", func(t *testing.T) {
		item := &data.Item{ItemId: "a", Labels: map[string]any{"embedding": []any{0.5, 1.0}}}
		embedding, present, err := ItemEmbedding(cfg, item)
		require.NoError(t, err)
		assert.True(t, present)
		assert.Equal(t, []float32{0.5, 1}, embedding)
	})
	t.Run("missing column", func(t *testing.T) {
		item := &data.Item{ItemId: "a", Labels: map[string]any{"title": "x"}}
		_, present, err := ItemEmbedding(cfg, item)
		require.NoError(t, err)
		assert.False(t, present)
	})
	t.Run("nil labels", func(t *testing.T) {
		_, present, err := ItemEmbedding(cfg, &data.Item{ItemId: "a"})
		require.NoError(t, err)
		assert.False(t, present)
	})
	t.Run("non numeric column", func(t *testing.T) {
		item := &data.Item{ItemId: "a", Labels: map[string]any{"embedding": []any{"x", "y"}}}
		_, present, err := ItemEmbedding(cfg, item)
		assert.Error(t, err)
		assert.True(t, present)
	})
	t.Run("non embedding recommender", func(t *testing.T) {
		item := &data.Item{ItemId: "a", Labels: map[string]any{"embedding": []any{0.5, 1.0}}}
		_, present, err := ItemEmbedding(config.ItemToItemConfig{Name: "tags", Type: "tags", Column: "item.Labels.tags"}, item)
		require.NoError(t, err)
		assert.False(t, present)
	})
}

func TestIndexItemVector(t *testing.T) {
	ctx := t.Context()
	vectorClient, _ := newIncrementalFixture(t)
	cfg := config.ItemToItemConfig{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"}
	now := time.Now()

	// first write creates the collection
	indexed, err := IndexItemVector(ctx, vectorClient, cfg, vectors.VectorConfig{}, &data.Item{
		ItemId: "a", Labels: map[string]any{"embedding": []float32{1, 0}}, Categories: []string{"movie"},
	}, now)
	require.NoError(t, err)
	assert.True(t, indexed)
	info, err := vectorClient.DescribeCollection(ctx, vectors.ItemToItemCollection("neighbors"))
	require.NoError(t, err)
	assert.Equal(t, 2, info.Dimension)
	assert.Equal(t, vectors.Euclidean, info.Distance)

	// items without embeddings are skipped without error
	indexed, err = IndexItemVector(ctx, vectorClient, cfg, vectors.VectorConfig{}, &data.Item{ItemId: "b"}, now)
	require.NoError(t, err)
	assert.False(t, indexed)

	// dimension mismatches are refused
	_, err = IndexItemVector(ctx, vectorClient, cfg, vectors.VectorConfig{}, &data.Item{
		ItemId: "c", Labels: map[string]any{"embedding": []float32{1, 0, 0}},
	}, now)
	assert.ErrorIs(t, err, ErrEmbeddingDimensionMismatch)

	// writes are upserts
	indexed, err = IndexItemVector(ctx, vectorClient, cfg, vectors.VectorConfig{}, &data.Item{
		ItemId: "a", Labels: map[string]any{"embedding": []float32{0, 1}}, IsHidden: true,
	}, now)
	require.NoError(t, err)
	assert.True(t, indexed)
	stored, err := vectorClient.GetVectors(ctx, vectors.ItemToItemCollection("neighbors"), []string{"a"})
	require.NoError(t, err)
	require.Len(t, stored, 1)
	assert.Equal(t, []float32{0, 1}, stored[0].Values)
	count, err := vectorClient.CountVectors(ctx, vectors.ItemToItemCollection("neighbors"))
	require.NoError(t, err)
	assert.Equal(t, int64(1), count)
}

func TestQueryItemToItemWithFallback(t *testing.T) {
	ctx := t.Context()
	vectorClient, dataClient := newIncrementalFixture(t)
	cfg := config.ItemToItemConfig{Name: "neighbors", Type: "embedding", Column: "item.Labels.embedding"}
	now := time.Now()
	collection := vectors.ItemToItemCollection("neighbors")

	// two indexed items and one item that only exists in the data store
	require.NoError(t, vectorClient.AddCollection(ctx, collection, 2, vectors.Euclidean, vectors.VectorConfig{}))
	require.NoError(t, vectorClient.AddVectors(ctx, collection, []vectors.Vector{
		{Id: "near", Values: []float32{0.1, 0}, Categories: []string{"movie"}, Timestamp: now},
		{Id: "far", Values: []float32{10, 0}, Categories: []string{"movie"}, Timestamp: now},
	}))
	require.NoError(t, dataClient.BatchInsertItems(ctx, []data.Item{
		{ItemId: "new", Labels: map[string]any{"embedding": []float32{0, 0}}, Timestamp: now},
		{ItemId: "near", Labels: map[string]any{"embedding": []float32{0.1, 0}}, Timestamp: now},
		{ItemId: "plain", Timestamp: now},
	}))

	// fallback disabled: nothing indexed for the new item
	scores, usedFallback, err := QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "new", nil, 10, false)
	require.NoError(t, err)
	assert.False(t, usedFallback)
	assert.Empty(t, scores)

	// fallback enabled: neighbors come from the stored embedding
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "new", nil, 10, true)
	require.NoError(t, err)
	assert.True(t, usedFallback)
	assert.Equal(t, []string{"near", "far"}, lo.Map(scores, func(score cache.Score, _ int) string { return score.Id }))
	fallbackScores := scores

	// once indexed, the indexed path must produce exactly the same scores
	indexed, err := IndexItemVector(ctx, vectorClient, cfg, vectors.VectorConfig{},
		&data.Item{ItemId: "new", Labels: map[string]any{"embedding": []float32{0, 0}}}, now)
	require.NoError(t, err)
	assert.True(t, indexed)
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "new", nil, 10, true)
	require.NoError(t, err)
	assert.False(t, usedFallback)
	assert.Equal(t, fallbackScores, scores)

	// indexed items keep using the indexed path and never return themselves
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "near", nil, 10, true)
	require.NoError(t, err)
	assert.False(t, usedFallback)
	assert.Equal(t, []string{"new", "far"}, lo.Map(scores, func(score cache.Score, _ int) string { return score.Id }))

	// items without an embedding and unknown items have no neighbors
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "plain", nil, 10, true)
	require.NoError(t, err)
	assert.True(t, usedFallback)
	assert.Empty(t, scores)
	scores, _, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient, cfg, "missing", nil, 10, true)
	require.NoError(t, err)
	assert.Empty(t, scores)

	// a missing collection is an empty result while fallback is enabled
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient,
		config.ItemToItemConfig{Name: "other", Type: "embedding", Column: "item.Labels.embedding"}, "new", nil, 10, true)
	require.NoError(t, err)
	assert.True(t, usedFallback)
	assert.Empty(t, scores)
	scores, _, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient,
		config.ItemToItemConfig{Name: "other", Type: "embedding", Column: "item.Labels.embedding"}, "new", nil, 10, false)
	require.NoError(t, err, "a missing collection is empty even without the fallback")
	assert.Empty(t, scores)

	// tags recommenders never fall back
	scores, usedFallback, err = QueryItemToItemWithFallback(ctx, vectorClient, dataClient,
		config.ItemToItemConfig{Name: "neighbors", Type: "tags", Column: "item.Labels.tags"}, "new", nil, 10, true)
	require.NoError(t, err)
	assert.False(t, usedFallback)
	assert.Empty(t, scores)
}

func TestVectorWriterStats(t *testing.T) {
	ctx := t.Context()
	vectorClient, _ := newIncrementalFixture(t)
	writer := newSimilarityVectorWriter(ctx, vectorClient, "stats", vectors.Euclidean, vectors.VectorConfig{}, time.Now(), 10, false)
	require.NoError(t, writer.Add(vectors.Vector{Id: "a", Values: []float32{1, 0}}))
	require.NoError(t, writer.Add(vectors.Vector{Id: "b", Values: []float32{0, 1}}))
	require.NoError(t, writer.Add(vectors.Vector{Id: "c", Values: []float32{0, 1, 1}}))
	require.NoError(t, writer.Add(vectors.Vector{Id: "d"}))
	require.NoError(t, writer.Clean())
	added, skipped, invalid := writer.VectorWriterStats()
	assert.Equal(t, 3, added)
	assert.Equal(t, 1, skipped)
	assert.Equal(t, 1, invalid)
}

func TestQueryItemToItemMissingCollection(t *testing.T) {
	vectorClient, _ := newIncrementalFixture(t)
	cfg := config.ItemToItemConfig{Name: "unbuilt", Type: "embedding", Column: "item.Labels.embedding"}
	scores, err := QueryItemToItem(t.Context(), vectorClient, cfg, "any", nil, 10)
	require.NoError(t, err, "a recommender without a collection has no neighbors instead of failing the caller")
	assert.Empty(t, scores)
}

func TestAppendSparseVectorDeduplicates(t *testing.T) {
	idf := []float32{1, 1, 1, 1}
	vector := newSparseVector([]int32{0, 1, 1, 3, 3, 3}, idf, 0)
	assert.Equal(t, []uint32{0, 1, 3}, vector.Indices)
	assert.Len(t, vector.Values, 3)
}
