"""``SimpleNUTS``: the full-trajectory-in-memory NUTS, kept as a reference oracle.

This stores every leaf of each subtree in a fixed-size buffer and performs the subtree
U-turn checks by indexing the stored leaves directly. It is simpler and obviously
correct, at O(2^max_tree_depth) working memory. The memory-efficient :class:`NUTS`
(``nuts.py``) is the version to use; ``SimpleNUTS`` exists so the efficient one can be
checked against it (they trace the identical trajectory given the same RNG).

Only ``_build_subtree`` differs from :class:`~mimcs.hmc.nuts.BaseNUTS`; everything else
(outer doubling loop, U-turn, multinomial selection, reversible divergence test, kernel,
diagnostics) is inherited.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .nuts import (
    BaseNUTS, NUTSTree, _tree_index, _tree_set, _leaf_proxy_accept, _leaf_grad_evals)


class SimpleNUTS(BaseNUTS):
    """Reference NUTS that stores each subtree's leaves in full (O(2^max_tree_depth))."""

    def _build_subtree(self, frontier, eps, depth, H0, leaf_u, leaf_ls, ctx):
        dim = frontier.q.shape[0]
        emits = self.integrator.emits_step_size_proxy
        n_total = jnp.left_shift(jnp.int32(1), depth)       # 2^depth (traced depth)
        offset = n_total - 1        # this depth's slice of the flat leaf draws (mirrors NUTS)
        buf = jax.tree.map(
            lambda x: jnp.zeros((self._max_subtree,) + x.shape, x.dtype), frontier)
        psum_prefix = jnp.zeros((self._max_subtree, dim))

        def cond(c):
            n, *_, turning, diverging = c
            return (n < n_total) & (~turning) & (~diverging)

        def body(c):
            (n, frontier, buf, psum_prefix, leaf0, proposal, sub_logw, sub_psum,
             h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = c

            leaf = self.integrator.step(
                frontier, eps, ctx, None if leaf_ls is None else leaf_ls[offset + n])
            grad_evals_leaf = _leaf_grad_evals(frontier, leaf)
            H = self.total_energy(leaf, ctx)
            logw = self._leaf_log_weight(H, H0) + leaf.log_weight   # integrator correction
            a_leaf = jnp.where(jnp.isfinite(H), jnp.minimum(1.0, jnp.exp(H0 - H)), 0.0)

            buf = _tree_set(buf, n, leaf)
            prev = jnp.where(n > 0, psum_prefix[jnp.maximum(n - 1, 0)], jnp.zeros(dim))
            cur_psum = prev + leaf.p
            psum_prefix = psum_prefix.at[n].set(cur_psum)
            leaf0 = jax.tree.map(lambda a, b: jnp.where(n == 0, b, a), leaf0, leaf)

            new_logw = jnp.logaddexp(sub_logw, logw)
            take = jnp.log(leaf_u[offset + n]) < (logw - new_logw)
            proposal = jax.tree.map(lambda a, b: jnp.where(take, b, a), proposal, leaf)
            sub_logw = new_logw
            sub_psum = sub_psum + leaf.p

            h_min = jnp.minimum(h_min, H)
            h_max = jnp.maximum(h_max, H)
            diverging = (diverging | (~jnp.isfinite(H)) | (~jnp.isfinite(leaf.log_weight))
                         | ((h_max - h_min) > self.divergence_threshold))
            sum_accept = sum_accept + a_leaf
            sum_proxy_accept = sum_proxy_accept + _leaf_proxy_accept(emits, leaf)
            sum_grad_evals = sum_grad_evals + grad_evals_leaf

            def check(i, turn):
                size = jnp.left_shift(jnp.int32(1), i)
                a = (n + 1) - size
                closes = (jnp.bitwise_and(n + 1, size - 1) == 0) & (a >= 0) & (i <= depth)
                end_a = _tree_index(buf, jnp.maximum(a, 0))
                rho = cur_psum - jnp.where(
                    a > 0, psum_prefix[jnp.maximum(a - 1, 0)], jnp.zeros(dim))
                t = self._is_turning(end_a, leaf, rho, ctx)
                return jnp.where(closes, turn | t, turn)

            turning = jax.lax.fori_loop(1, self.max_tree_depth + 1, check, turning)
            return (n + 1, leaf, buf, psum_prefix, leaf0, proposal, sub_logw, sub_psum,
                    h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals,
                    turning, diverging)

        init = (jnp.int32(0), frontier, buf, psum_prefix, frontier, frontier,
                jnp.asarray(-jnp.inf), jnp.zeros(dim),
                jnp.asarray(jnp.inf), jnp.asarray(-jnp.inf), jnp.zeros(()),   # own range: h_min/h_max
                jnp.zeros(()), jnp.zeros(()), jnp.asarray(False), jnp.asarray(False))
        (n_final, last_leaf, _, _, leaf0, proposal, sub_logw, sub_psum,
         h_min, h_max, sum_accept, sum_proxy_accept, sum_grad_evals, turning, diverging) = \
            jax.lax.while_loop(cond, body, init)

        return NUTSTree(
            left=leaf0, right=last_leaf, proposal=proposal, momentum_sum=sub_psum,
            log_weight=sub_logw, h_min=h_min, h_max=h_max, sum_accept=sum_accept,
            sum_proxy_accept=sum_proxy_accept, sum_grad_evals=sum_grad_evals,
            n_leaves=n_final, depth=jnp.int32(0),
            terminated=turning | diverging, diverging=diverging)
