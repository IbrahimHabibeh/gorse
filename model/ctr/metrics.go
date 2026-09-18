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

// VideoHub fork: make silently skipped inference inputs observable.

package ctr

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// SkippedEmbeddingsTotal counts item embeddings ignored at inference time.
// "unknown_name" means the embedding column was not seen during training and
// "dimension_mismatch" means its length differs from the trained dimension.
// Both are zero-filled instead of panicking; a growing counter indicates that
// the catalog and the model disagree about the embedding schema.
var SkippedEmbeddingsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "gorse",
	Subsystem: "ctr",
	Name:      "skipped_embeddings_total",
	Help:      "Item embeddings ignored during click-through rate inference by reason.",
}, []string{"reason"})
