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

// VideoHub fork: memory telemetry for cache backends that can report it.

package cache

import (
	"context"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

// MemoryReporter is implemented by cache stores that can report their memory
// usage in bytes. It is optional so that backends without the information do
// not have to change.
type MemoryReporter interface {
	MemoryUsage(ctx context.Context) (int64, error)
}

// MemoryUsage returns Redis' used_memory from INFO memory.
func (r *Redis) MemoryUsage(ctx context.Context) (int64, error) {
	info, err := r.client.Info(ctx, "memory").Result()
	if err != nil {
		return 0, errors.WithStack(err)
	}
	return parseRedisUsedMemory(info)
}

func parseRedisUsedMemory(info string) (int64, error) {
	for _, line := range strings.Split(info, "\n") {
		line = strings.TrimSpace(line)
		if value, ok := strings.CutPrefix(line, "used_memory:"); ok {
			used, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			if err != nil {
				return 0, errors.WithStack(err)
			}
			return used, nil
		}
	}
	return 0, errors.New("used_memory not found in INFO memory")
}
