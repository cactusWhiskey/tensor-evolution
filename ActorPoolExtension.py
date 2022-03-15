import ray
from ray.util import ActorPool


class ActorPoolExtension(ActorPool):

    def map_ordered_return_all(self, fn, values) -> list:
        """Apply the given function in parallel over the actors and values.

        This returns a list.

        Arguments:
            fn (func): Function that takes (actor, value) as argument and
                returns an ObjectRef computing the result over the value. The
                actor will be considered busy until the ObjectRef completes.
            values (list): List of values that fn(actor, value) should be
                applied to.

        Returns:
            List of results from applying fn to the actors and values.


        """

        results = [None] * len(values)

        for index, v in enumerate(values):
            self.submit(fn, v, original_index=index)
        while self.has_next():
            result, j = self.get_next_unordered_with_index()
            results[j] = result
        return results

    def submit(self, fn, value, original_index=None):
        """Schedule a single task to run in the pool.

        This has the same argument semantics as map(), but takes on a single
        value instead of a list of values. The result can be retrieved using
        get_next() / get_next_unordered().

        Arguments:
            fn (func): Function that takes (actor, value) as argument and
                returns an ObjectRef computing the result over the value. The
                actor will be considered busy until the ObjectRef completes.
            value (object): Value to compute a result for.
            original_index: value's position index from the iterable of values

        Examples:
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            >>> print(pool.get_next(), pool.get_next())
            2, 4
        """
        if self._idle_actors:
            actor = self._idle_actors.pop()
            future = fn(actor, value)
            future_key = tuple(future) if isinstance(future, list) else future
            self._future_to_actor[future_key] = (self._next_task_index, actor, original_index)
            self._index_to_future[self._next_task_index] = future
            self._next_task_index += 1
        else:
            self._pending_submits.append((fn, value, original_index))

    def get_next(self, timeout=None):
        """Returns the next pending result in order.

        This returns the next result produced by submit(), blocking for up to
        the specified timeout until it is available.

        Returns:
            The next result.

        Raises:
            TimeoutError if the timeout is reached.

        Examples:
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> print(pool.get_next())
            2
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        if self._next_return_index >= self._next_task_index:
            raise ValueError("It is not allowed to call get_next() after "
                             "get_next_unordered().")
        future = self._index_to_future[self._next_return_index]
        if timeout is not None:
            res, _ = ray.wait([future], timeout=timeout)
            if not res:
                raise TimeoutError("Timed out waiting for result")
        del self._index_to_future[self._next_return_index]
        self._next_return_index += 1

        future_key = tuple(future) if isinstance(future, list) else future
        i, a, original_index = self._future_to_actor.pop(future_key)

        self._return_actor(a)
        return ray.get(future)

    def get_next_unordered(self, timeout=None):
        """Returns any of the next pending results.

        This returns some result produced by submit(), blocking for up to
        the specified timeout until it is available. Unlike get_next(), the
        results are not always returned in same order as submitted, which can
        improve performance.

        Returns:
            The next result.

        Raises:
            TimeoutError if the timeout is reached.

        Examples:
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            >>> print(pool.get_next_unordered())
            4
            >>> print(pool.get_next_unordered())
            2
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        # TODO(ekl) bulk wait for performance
        res, _ = ray.wait(
            list(self._future_to_actor), num_returns=1, timeout=timeout)
        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")
        i, a, original_index = self._future_to_actor.pop(future)
        self._return_actor(a)
        del self._index_to_future[i]
        self._next_return_index = max(self._next_return_index, i + 1)
        return ray.get(future)

    def get_next_unordered_with_index(self, timeout=None):
        """Returns any of the next pending results, with its positional index from
                the original iterable of value.

        This returns some result produced by submit(), blocking for up to
        the specified timeout until it is available. Unlike get_next(), the
        results are not always returned in same order as submitted, which can
        improve performance.

        Returns:
            The next result, with index. Tuple (result, index)

        Raises:
            TimeoutError if the timeout is reached.


        """

        if not self.has_next():
            raise StopIteration("No more results to get")
        # TODO(ekl) bulk wait for performance
        res, _ = ray.wait(
            list(self._future_to_actor), num_returns=1, timeout=timeout)
        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")
        i, a, original_index = self._future_to_actor.pop(future)
        self._return_actor(a)
        del self._index_to_future[i]
        self._next_return_index = max(self._next_return_index, i + 1)
        return ray.get(future), original_index

