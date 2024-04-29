from typing import Dict, List, Tuple

import numpy as np
import torch
import wigners
from metatensor.torch import Labels, TensorBlock, TensorMap
import mops.torch

from .tensor_sum import TensorAdd
from .linear import Linear


class CGIterator(torch.nn.Module):
    def __init__(
        self,
        k_max_l: List[int],
        number_of_iterations,
        cgs,
        irreps_in,
        exponential_algorithm=True,
        requested_LS_string=None,
    ):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.number_of_iterations = number_of_iterations
        self.cgs = cgs
        self.requested_LS_string = requested_LS_string

        cg_iterations = []
        for n_iteration in range(self.number_of_iterations):
            if n_iteration == self.number_of_iterations - 1:
                requested_LS_string_now = requested_LS_string
            else:
                requested_LS_string_now = None
            if n_iteration == 0:
                irreps_in_1 = irreps_in
            else:
                irreps_in_1 = irreps_out
            irreps_in_2 = irreps_in_1 if exponential_algorithm else irreps_in
            cg_iterations.append(
                CGIterationAdd(
                    self.k_max_l, cgs, irreps_in_1, irreps_in_2, requested_LS_string_now
                )
            )
            irreps_out = cg_iterations[-1].irreps_out
        self.cg_iterations = torch.nn.ModuleList(cg_iterations)
        self.exponential_algorithm = exponential_algorithm
        self.irreps_out = cg_iterations[-1].irreps_out

    def forward(self, density: TensorMap):
        starting_density = density
        current_density = density
        for iterator in self.cg_iterations:
            if self.exponential_algorithm:
                current_density = iterator(current_density, current_density)
            else:
                current_density = iterator(current_density, starting_density)
        return current_density


class CGIterationAdd(torch.nn.Module):
    def __init__(
        self,
        k_max_l: List[int],
        cgs,
        irreps_in_1,
        irreps_in_2,
        requested_LS_string=None,
    ):
        super().__init__()
        self.l_max = len(k_max_l) - 1
        self.cg_iteration = CGIteration(
            k_max_l, irreps_in_1, irreps_in_2, cgs, requested_LS_string
        )
        self.irreps_out = self.cg_iteration.irreps_out
        discard = False if requested_LS_string is None else True
        self.adder = TensorAdd(discard)

    def forward(self, features_1: TensorMap, features_2: TensorMap):
        features_out = self.cg_iteration(features_1, features_2)
        features_out = (
            self.adder(features_1, features_out)
            if self.adder.discard
            else self.adder(features_out, features_1)
        )
        return features_out


class CGIteration(torch.nn.Module):

    def __init__(
        self,
        k_max_l: List[int],
        irreps_in_1: List[Tuple[int, int]],
        irreps_in_2: List[Tuple[int, int]],
        cgs,
        requested_LS_string=None,
    ):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.cgs = cgs
        self.irreps_out = []
        self.requested_LS_string = requested_LS_string

        self.sizes_by_lam_sig: Dict[str, int] = {}
        for l1, s1 in irreps_in_1:
            for l2, s2 in irreps_in_2:
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    S = s1 * s2 * (-1) ** (l1 + l2 + L)
                    if self.requested_LS_string is not None:
                        if str(L) + "_" + str(S) != self.requested_LS_string:
                            continue
                    if (L, S) not in self.irreps_out:
                        self.irreps_out.append((L, S))
                    larger_l = max(l1, l2)
                    size = self.k_max_l[larger_l]
                    if (str(L) + "_" + str(S)) not in self.sizes_by_lam_sig:
                        self.sizes_by_lam_sig[(str(L) + "_" + str(S))] = size
                    else:
                        self.sizes_by_lam_sig[(str(L) + "_" + str(S))] += size

        # Register linear layers for contraction:
        self.linear_contractions = torch.nn.ModuleDict(
            {
                LS_string: Linear(size_LS, k_max_l[int(LS_string.split("_")[0])])
                for LS_string, size_LS in self.sizes_by_lam_sig.items()
            }
        )
        values_out = [
            [int(LS_string.split("_")[0]), int(LS_string.split("_")[1])]
            for LS_string in list(self.sizes_by_lam_sig.keys())
        ]
        self.keys_out = Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor(values_out, dtype=torch.long),
        )

    def forward(self, features_1: TensorMap, features_2: TensorMap):
        # handle dtype and device of the cgs
        if self.cgs["0_0"]["indices_a"].device != features_1.device:
            self.cgs = {
                key: {
                    key2: value2.to(device=features_1.device)
                    for key2, value2 in value.items()
                }
                for key, value in self.cgs.items()
            }
        if self.cgs["0_0"]["indices_a"].dtype != features_1.dtype:
            self.cgs = {
                key: {
                    key2: (value2.to(dtype=features_1.dtype) if key2 == "C" else value2)
                    for key2, value2 in value.items()
                }
                for key, value in self.cgs.items()
            }

        # COULD DECREASE COST IF SYMMETRIC
        # Assume first and last dimension is the same for both
        results_by_lam_sig: Dict[str, List[torch.Tensor]] = {}
        for key_ls_1, block_ls_1 in features_1.items():
            l1s1 = key_ls_1.values
            l1, s1 = int(l1s1[0]), int(l1s1[1])
            for key_ls_2, block_ls_2 in features_2.items():
                l2s2 = key_ls_2.values
                l2, s2 = int(l2s2[0]), int(l2s2[1])
                min_size = min(block_ls_1.values.shape[2], block_ls_2.values.shape[2])
                tensor1 = block_ls_1.values[:, :, :min_size]
                tensor2 = block_ls_2.values[:, :, :min_size]
                A = tensor1.swapaxes(1, 2).reshape(tensor1.shape[0] * min_size, tensor1.shape[1])
                B = tensor2.swapaxes(1, 2).reshape(tensor2.shape[0] * min_size, tensor2.shape[1])
                cgs = self.cgs[str(l1) + "_" + str(l2)]
                indices_a = cgs["indices_a"]
                indices_b = cgs["indices_b"]
                indices_output = cgs["indices_output"]
                C = cgs["C"]
                split_chunks = cgs["split_chunks"]
                size_output = torch.sum(split_chunks).item()
                
                result = mops.torch.sparse_accumulation_of_products(
                    A, B, C, indices_a, indices_b, indices_output, size_output
                )
                result = result.reshape(-1, min_size, size_output)
                result = result.swapaxes(1, 2)
                split_chunks_list: List[int] = split_chunks.tolist()
                split_result = torch.split(result, split_chunks_list, dim=1)
                for idx, L in enumerate(range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1)):
                    S = int(s1 * s2 * (-1) ** (l1 + l2 + L))
                    if self.requested_LS_string is not None:
                        if str(L) + "_" + str(S) != self.requested_LS_string:
                            continue
                    if (str(L) + "_" + str(S)) not in results_by_lam_sig:
                        results_by_lam_sig[(str(L) + "_" + str(S))] = [split_result[idx]]
                    else:
                        results_by_lam_sig[(str(L) + "_" + str(S))].append(split_result[idx])

        compressed_results_by_lam_sig: Dict[str, torch.Tensor] = {}
        for LS_string, linear_LS in self.linear_contractions.items():
            split_LS_string = LS_string.split("_")
            L = int(split_LS_string[0])
            S = int(split_LS_string[1])
            concatenated_tensor = torch.concatenate(
                results_by_lam_sig[LS_string], dim=2
            )
            compressed_tensor = linear_LS(concatenated_tensor)
            compressed_results_by_lam_sig[(str(L) + "_" + str(S))] = compressed_tensor

        blocks: List[TensorBlock] = []
        for LS_string, compressed_tensor_LS in compressed_results_by_lam_sig.items():
            split_LS_string = LS_string.split("_")
            L = int(split_LS_string[0])
            S = int(split_LS_string[1])
            blocks.append(
                TensorBlock(
                    values=compressed_tensor_LS,
                    samples=features_1.block({"o3_lambda": 0, "o3_sigma": 1}).samples,
                    components=[
                        Labels(
                            names=["mu"],
                            values=torch.arange(
                                start=-L,
                                end=L + 1,
                                dtype=torch.int,
                                device=compressed_tensor_LS.device,
                            ).reshape(2 * L + 1, 1),
                        ).to(compressed_tensor_LS.device)
                    ],
                    properties=Labels(
                        names=["properties"],
                        values=torch.arange(
                            compressed_tensor_LS.shape[2],
                            dtype=torch.int,
                            device=compressed_tensor_LS.device,
                        ).reshape(compressed_tensor_LS.shape[2], 1),
                    ),
                )
            )

        return TensorMap(keys=self.keys_out.to(blocks[0].values.device), blocks=blocks)


def cg_combine_l1l2L(tensor12, cg_tensor):
    out_tensor = tensor12 @ cg_tensor.reshape(
        cg_tensor.shape[0] * cg_tensor.shape[1], cg_tensor.shape[2]
    )
    return out_tensor.swapaxes(1, 2)


def get_cg_coefficients(l_max, sparse):
    cg_object = ClebschGordanReal(sparse)
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                cg_object._add(l1, l2, L)

    if sparse:
        cg_object.stack_sparse(l_max)

    return cg_object


class ClebschGordanReal:

    def __init__(self, sparse):
        self._cgs = {}
        self.sparse = sparse

    def _add(self, l1, l2, L):
        # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

        if self._cgs is None:
            raise ValueError("Trying to add CGs when not initialized... exiting")

        if (l1, l2, L) in self._cgs:
            raise ValueError("Trying to add CGs that are already present... exiting")

        maxx = max(l1, max(l2, L))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )

        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(real_cg.shape)
        real_cg = real_cg.swapaxes(0, 1)

        real_cg = real_cg @ c2r[L].T

        if (l1 + l2 + L) % 2 == 0:
            rcg = np.real(real_cg)
        else:
            rcg = np.imag(real_cg)

        # Zero any possible (and very rare) near-zero elements
        where_almost_zero = np.where(
            np.logical_and(np.abs(rcg) > 0, np.abs(rcg) < 1e-14)
        )
        if len(where_almost_zero[0] != 0):
            print("INFO: Found almost-zero CG!")
        for i0, i1, i2 in zip(
            where_almost_zero[0], where_almost_zero[1], where_almost_zero[2]
        ):
            rcg[i0, i1, i2] = 0.0

        if self.sparse:
            where_nonzero = np.where(rcg != 0)
            self._cgs[(l1, l2, L)] = {
                "indices_a": torch.tensor(where_nonzero[0], dtype=torch.int32),
                "indices_b": torch.tensor(where_nonzero[1], dtype=torch.int32),
                "indices_output": torch.tensor(where_nonzero[2], dtype=torch.int32),
                "C": torch.tensor(rcg[where_nonzero]).type(torch.get_default_dtype()),
            }

        else:
            self._cgs[(l1, l2, L)] = (
                torch.tensor(rcg).type(torch.get_default_dtype())
            )

    def get(self, key):
        if key in self._cgs:
            return self._cgs[key]
        else:
            self._add(key[0], key[1], key[2])
            return self._cgs[key]
        
    def stack_sparse(self, l_max):
        if not self.sparse:
            raise ValueError("Cannot stack sparse tensors if not sparse")

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                indices_a = []
                indices_b = []
                indices_output = []
                C = []
                split_chunks = []
                current_output_shift = 0
                for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                    if (l1, l2, L) not in self._cgs:
                        raise ValueError(
                            f"CGs for l1={l1}, l2={l2}, L={L} not found... exiting"
                        )
                    cgs = self._cgs.pop((l1, l2, L))
                    indices_a.append(cgs["indices_a"])
                    indices_b.append(cgs["indices_b"])
                    indices_output.append(cgs["indices_output"] + current_output_shift)
                    C.append(cgs["C"])
                    split_chunks.append(2*L + 1)
                    current_output_shift += 2 * L + 1
                self._cgs[(l1, l2)] = {
                    "indices_a": torch.cat(indices_a),
                    "indices_b": torch.cat(indices_b),
                    "indices_output": torch.cat(indices_output),
                    "C": torch.cat(C),
                    "split_chunks": torch.tensor(split_chunks),
                }             


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)